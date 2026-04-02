"""
Dynamic Graph Condensation for Large-Scale Graphs.

This script performs graph condensation on large dynamic graphs that cannot
fit entirely in GPU memory. Key differences from small-scale condensation:

1. Uses subgraph sampling (via subgraph_extracter.py) to create mini-batches
2. Uses efficient MMD loss with precomputed bandwidth and k(X,X)
3. Features are precomputed on CPU to avoid OOM
4. Teacher/val models operate on sampled subgraphs for evaluation

Workflow:
1. Run subgraph_extracter.py first to create sampled batches
2. Run this script for condensation
3. Run test_large.py to evaluate
"""
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gc
import math
import random
import argparse
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import deeprobust.graph.utils as utils
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score

from src.models.DGNN import GCRN, TGCN, TGCN_L, DySAT, STGCN, ROLAND
from src.models.basicgnn import propagater
from src.models.structure_generation import SNN_generator
from src.utils import gcn_norm, RBF_eff, MMDLoss_eff
from src.utils.graph_utils import GraphData  # Required for torch.load

warnings.filterwarnings("ignore")


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Graph Condensation (Large Scale)')
    
    # Basic settings
    parser.add_argument('--cuda', type=int, default=0, help='GPU device id')
    parser.add_argument('--dataset', type=str, default='arxiv', help='Dataset name')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    
    # Model architecture
    parser.add_argument('--nlayers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--K', type=int, default=3, help='Propagation depth for feature aggregation')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--teacher_model', type=str, default='TGCN_L',
                       choices=['GCRN', 'TGCN', 'TGCN_L', 'DySAT', 'STGCN', 'ROLAND'])
    parser.add_argument('--val_model', type=str, default='TGCN_L',
                       choices=['GCRN', 'TGCN', 'TGCN_L', 'DySAT', 'STGCN', 'ROLAND'])
    
    # Condensation settings
    parser.add_argument('--reduction_rate', type=float, default=0.005, help='Graph reduction rate')
    parser.add_argument('--lr_adj', type=float, default=0.05, help='Learning rate for adjacency')
    parser.add_argument('--lr_feat', type=float, default=0.05, help='Learning rate for features')
    parser.add_argument('--lr_model', type=float, default=0.001, help='Learning rate for model')
    parser.add_argument('--lr_teacher_model', type=float, default=0.005, help='Learning rate for teacher')
    
    # Loss settings
    parser.add_argument('--loss_factor', type=float, default=10, help='MMD loss weight')
    parser.add_argument('--temporal_alpha', type=float, default=0.5, help='Temporal smoothing factor')
    
    # Training loops
    parser.add_argument('--teacher_training_loop', type=int, default=100, help='Teacher training epochs')
    parser.add_argument('--condensing_loop', type=int, default=1000, help='Condensation iterations')
    parser.add_argument('--condensing_val_stage', type=int, default=100, help='Validation frequency')
    parser.add_argument('--student_model_loop', type=int, default=2000, help='Student training epochs')
    parser.add_argument('--student_val_stage', type=int, default=200, help='Student validation frequency')
    
    return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def generate_labels_syn(labels_train, num_classes, reduction_rate):
    """Generate synthetic labels based on class distribution."""
    from collections import Counter
    counter = Counter(labels_train.cpu().numpy())
    num_class_dict = {}
    
    sorted_counter = sorted(counter.items(), key=lambda x: x[1])
    labels_syn = []
    
    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * reduction_rate)
        labels_syn += [c] * num_class_dict[c]
    
    return labels_syn, num_class_dict


def gcn_norm_temporal(adjs, nnodes):
    """Apply GCN normalization to temporal adjacency matrices."""
    edge_indexs = []
    edge_weights = []
    for adj in adjs:
        edge_index = torch.nonzero(adj).T
        edge_weight = adj[edge_index[0], edge_index[1]]
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, nnodes)
        edge_indexs.append(edge_index)
        edge_weights.append(edge_weight)
    return edge_indexs, edge_weights


def to_device(batch, device='cuda:0'):
    """Move batch data to device."""
    return {
        "nodes": batch["nodes"].to(device),
        "Feats": [feat.to(device) for feat in batch["Feats"]],
        "Adjs": [adj.to(device) for adj in batch["Adjs"]],
        "mask": batch["mask"].to(device)
    }


def test_on_batches(input_batch, model, labels, device):
    """Evaluate model on batched data."""
    Logits = []
    Labels = []
    for batch in input_batch:
        batch = to_device(batch, device)
        output = model.predict(feats=batch["Feats"], adjs=batch["Adjs"], masks=batch["mask"])
        Logits.append(output)
        Labels.append(labels[batch["nodes"]])
    Logits = torch.cat(Logits, dim=0).argmax(1)
    Labels = torch.cat(Labels, dim=0)
    micro_f1 = f1_score(Logits.cpu(), Labels.cpu(), average='micro')
    macro_f1 = f1_score(Logits.cpu(), Labels.cpu(), average='macro')
    return micro_f1, macro_f1


def get_model(model_name, d, hidden, nclass, num_steps, nlayers, dropout, device):
    """Initialize a DGNN model by name."""
    model_map = {
        'GCRN': lambda: GCRN(in_features=d, hidden_feature=hidden, out_features=nclass,
                            num_timesteps=num_steps, nlayers=nlayers, dropout=dropout),
        'TGCN': lambda: TGCN(in_features=d, hidden_feature=hidden, out_features=nclass,
                            num_timesteps=num_steps, nlayers=nlayers, dropout=dropout),
        'TGCN_L': lambda: TGCN_L(in_features=d, hidden_feature=hidden, out_features=nclass,
                                num_timesteps=num_steps, nlayers=nlayers, dropout=dropout),
        'DySAT': lambda: DySAT(in_features=d, hidden_feature=hidden, out_features=nclass,
                              num_timesteps=num_steps, nlayers=nlayers, dropout=dropout),
        'STGCN': lambda: STGCN(in_features=d, hidden_feature=hidden, out_features=nclass,
                              num_timesteps=num_steps, nlayers=nlayers, dropout=dropout),
        'ROLAND': lambda: ROLAND(in_features=d, hidden_feature=hidden, out_features=nclass,
                                num_timesteps=num_steps, nlayers=nlayers, dropout=dropout),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    return model_map[model_name]().to(device)


# ============================================================================
# Training Functions
# ============================================================================
class LargeCondenser:
    """Main class for large-scale dynamic graph condensation."""
    
    def __init__(self, args, feats, adjs, global_adjs, labels, 
                 idx_train, idx_val, idx_test, 
                 train_batch, val_batch, test_batch, device):
        self.args = args
        self.device = device
        self.root = os.path.abspath(os.path.dirname(__file__))
        
        # Data (feats and adjs are on CPU for large graphs)
        self.feats = feats  # CPU
        self.adjs = adjs    # CPU edge indices
        self.global_adjs = global_adjs  # CPU SparseTensors for propagation
        self.labels = labels  # GPU
        
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        
        # Sampled batches for training/evaluation
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        
        # Derived info
        self.num_steps = feats.shape[0]
        self.n_nodes = feats.shape[1]
        self.d = feats.shape[-1]
        self.nclass = int(labels.max() + 1)
        
        self.labels_train = labels[idx_train]
        self.feat_train = feats[:, idx_train, :]  # CPU
        
        # Initialize synthetic data
        self._init_synthetic_data()
        
    def _init_synthetic_data(self):
        """Initialize synthetic features and labels."""
        labels_syn, num_class_dict = generate_labels_syn(
            self.labels_train, self.nclass, self.args.reduction_rate
        )
        self.labels_syn = torch.LongTensor(labels_syn).to(self.device)
        self.num_class_dict = num_class_dict
        self.nnodes_syn = len(self.labels_syn)
        
        # Initialize synthetic features
        self.feat_syn = nn.Parameter(
            torch.FloatTensor(self.num_steps, self.nnodes_syn, self.d).to(self.device)
        )
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        
        # Initialize structure generator
        self.structure_generator = SNN_generator(
            nfeat=self.d, nnodes=self.nnodes_syn, device=self.device,
            n_steps=self.num_steps, args=self.args
        ).to(self.device)
        
        print(f'Synthetic graph size: {self.feat_syn.shape}')
    
    def train_teacher(self, teacher_model):
        """Train the teacher model on sampled batches."""
        patience = 100
        stopping_counter = 0
        best_val = 0
        
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr=self.args.lr_teacher_model)
        
        for epoch in tqdm(range(1, self.args.teacher_training_loop + 1), desc="Training teacher"):
            for batch in self.train_batch:
                teacher_model.train()
                batch = to_device(batch, self.device)
                output = teacher_model.forward(feats=batch["Feats"], adjs=batch["Adjs"], masks=batch["mask"])
                optimizer.zero_grad()
                loss = F.nll_loss(output, self.labels[batch["nodes"]])
                loss.backward()
                optimizer.step()
            
            val_micro, val_macro = test_on_batches(self.val_batch, teacher_model, self.labels, self.device)
            test_micro, test_macro = test_on_batches(self.test_batch, teacher_model, self.labels, self.device)
            
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss.item():.4f}, '
                      f'Test micro: {100 * test_micro:.2f}%, Test macro: {100 * test_macro:.2f}%')
            
            if val_micro > best_val:
                best_val = val_micro
                stopping_counter = 0
                torch.save(
                    teacher_model.state_dict(),
                    f'{self.root}/teacher/{self.args.teacher_model}_{self.args.dataset}_{self.args.dropout}_{self.args.seed}.pt'
                )
            else:
                stopping_counter += 1
            
            if stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs!")
                break
    
    def condensation(self, teacher_model):
        """Main condensation loop with efficient MMD."""
        propagate = propagater().to('cpu')  # CPU for large graph propagation
        optimizer_feat = optim.Adam([self.feat_syn], lr=self.args.lr_feat)
        optimizer_pge = optim.Adam(self.structure_generator.parameters(), lr=self.args.lr_adj)
        
        # ===== Precompute features and MMD statistics (on CPU) =====
        print("Precomputing features and MMD statistics...")
        feats_org, coeff, coeff_sum, PreComputed_XX, bandwidths = self._precompute_features(propagate)
        
        # ===== Condensation loop =====
        best_val = 0
        best_test_micro = 0
        best_test_macro = 0
        loss_fn_MMD = MMDLoss_eff()
        propagate_gpu = propagater().to(self.device)
        alpha = self.args.temporal_alpha
        
        for i in tqdm(range(1, self.args.condensing_loop + 1), desc="Condensing"):
            # Generate synthetic adjacency
            adjs_syn = self.structure_generator(self.feat_syn)
            edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, self.nnodes_syn)
            
            # Compute aggregated features for synthetic data
            concat_feat_syn_all = []
            aggr_syn_memory = []
            for time_step in range(self.num_steps):
                concat_feat_syn = self.feat_syn[time_step]
                temp = self.feat_syn[time_step]
                edge_index_syn = edge_indexs_syn[time_step]
                edge_weight_syn = edge_weights_syn[time_step]
                
                for k in range(self.args.K):
                    aggr_syn = propagate_gpu(edge_index=edge_index_syn, x=temp, edge_weight=edge_weight_syn)
                    if time_step == 0:
                        aggr_syn_memory.append(aggr_syn)
                    else:
                        aggr_syn = alpha * aggr_syn_memory[k] + (1 - alpha) * aggr_syn
                        aggr_syn_memory[k] = aggr_syn
                    concat_feat_syn = torch.cat((concat_feat_syn, aggr_syn), dim=1)
                    temp = aggr_syn
                concat_feat_syn_all.append(concat_feat_syn)
            
            concat_feat_syn_all = torch.stack(concat_feat_syn_all)
            FEAT_syn = concat_feat_syn_all.transpose(0, 1)  # (N_syn, T, D')
            
            # Teacher loss
            teacher_model.train()
            output_syn = teacher_model.forward(self.feat_syn, edge_indexs_syn, edge_weights_syn)
            hard_loss = F.nll_loss(output_syn, self.labels_syn)
            
            # MMD loss with precomputed statistics
            feats_syn = []
            for c in range(self.nclass):
                if c in self.num_class_dict:
                    index = torch.where(self.labels_syn == c)[0]
                    feats_syn.append(FEAT_syn[index])
            
            mmd_loss = torch.tensor(0.0).to(self.device)
            c_idx = 0
            for c in range(self.nclass):
                if c in self.num_class_dict:
                    mmd_loss += coeff[c_idx] * loss_fn_MMD(
                        PreComputed_XX[c_idx], bandwidths[c_idx],
                        feats_org[c_idx], feats_syn[c_idx]
                    )
                    c_idx += 1
            mmd_loss = mmd_loss / coeff_sum
            
            # Total loss
            loss = hard_loss + self.args.loss_factor * mmd_loss
            
            # Update
            optimizer_feat.zero_grad()
            optimizer_pge.zero_grad()
            loss.backward()
            optimizer_pge.step()
            optimizer_feat.step()
            
            # Validation
            if i > 0 and i % self.args.condensing_val_stage == 0:
                best_val, best_test_micro, best_test_macro = self._validate(
                    teacher_model, best_val, best_test_micro, best_test_macro, i
                )
    
    def _precompute_features(self, propagate):
        """Precompute aggregated features and MMD statistics.
        
        This is done on CPU to avoid OOM for large graphs.
        """
        alpha = self.args.temporal_alpha
        concat_feat_all = []
        agg_memory = []
        
        # Propagate on full graph (CPU)
        for time_step in range(self.num_steps):
            concat_feat = self.feat_train[time_step]  # CPU
            temp = self.feats[time_step]  # CPU
            adj = self.global_adjs[time_step]  # CPU SparseTensor
            
            for k in range(self.args.K):
                aggr = propagate(edge_index=adj, x=temp, edge_weight=None)
                if time_step == 0:
                    agg_memory.append(aggr)
                else:
                    aggr = alpha * agg_memory[k] + (1 - alpha) * aggr
                    agg_memory[k] = aggr
                concat_feat = torch.cat((concat_feat, aggr[self.idx_train]), dim=1)
                temp = aggr
            concat_feat_all.append(concat_feat)
        
        concat_feat_all = torch.stack(concat_feat_all)
        FEAT_org = concat_feat_all.transpose(0, 1)  # (N_train, T, D')
        
        # Compute per-class statistics
        feats_org = []
        PreComputed_XX = []
        bandwidths = []
        coeff = []
        coeff_sum = 0
        kernel = RBF_eff()
        labels_train_cpu = self.labels_train.cpu()
        
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index = torch.where(labels_train_cpu == c)[0]
                coe = self.num_class_dict[c] / max(self.num_class_dict.values())
                coeff_sum += coe
                coeff.append(coe)
                
                feat_org_c = FEAT_org[index]  # CPU
                n_sample = feat_org_c.shape[0]
                
                # Compute bandwidth
                flat_X = feat_org_c.view(n_sample, -1)
                bandwidth = (torch.cdist(flat_X, flat_X) ** 2).data.sum() / (n_sample ** 2 - n_sample)
                bandwidths.append(bandwidth.to(self.device))
                
                # Precompute k(X, X)
                XX = kernel(bandwidth, feat_org_c, feat_org_c).mean()
                
                feats_org.append(feat_org_c.to(self.device))
                PreComputed_XX.append(XX.to(self.device))
        
        coeff_sum = torch.tensor(coeff_sum).to(self.device)
        return feats_org, coeff, coeff_sum, PreComputed_XX, bandwidths
    
    def _validate(self, teacher_model, best_val, best_test_micro, best_test_macro, epoch):
        """Validate synthetic data quality."""
        adjs_syn = self.structure_generator(self.feat_syn, inference=True)
        edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, self.nnodes_syn)
        
        # Create validation model
        val_model = get_model(
            self.args.val_model, self.d, self.args.hidden, self.nclass,
            self.num_steps, self.args.nlayers, self.args.dropout, self.device
        )
        optimizer = optim.AdamW(val_model.parameters(), lr=self.args.lr_model)
        val_model.initialize()
        
        # Check teacher accuracy on synthetic
        teacher_output_syn = teacher_model.predict(self.feat_syn, edge_indexs_syn, edge_weights_syn)
        acc = utils.accuracy(teacher_output_syn, self.labels_syn)
        print(f'Epoch {epoch} Teacher on syn accuracy= {100*acc.item():.2f}')
        
        # Train student
        for j in range(self.args.student_model_loop):
            val_model.train()
            output_syn = val_model.forward(self.feat_syn, edge_indexs_syn, edge_weights_syn)
            loss = F.nll_loss(output_syn, self.labels_syn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if j % self.args.student_val_stage == 0:
                val_micro, val_macro = test_on_batches(self.val_batch, val_model, self.labels, self.device)
                test_micro, test_macro = test_on_batches(self.test_batch, val_model, self.labels, self.device)
                
                if j % 400 == 0:
                    print(f'Epoch: {j:02d}, Loss: {loss.item():.4f}, '
                          f'Test Micro: {100 * test_micro:.2f}% Test Macro: {100 * test_macro:.2f}%')
                
                if val_micro >= best_val:
                    best_val = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    torch.save(self.feat_syn, f'{self.root}/syn/feat_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}.pt')
                    torch.save(adjs_syn, f'{self.root}/syn/adj_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}.pt')
                    torch.save(self.labels_syn, f'{self.root}/syn/label_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}.pt')
        
        print(f'Epoch: {epoch:02d}, Best Test Micro: {100 * best_test_micro:.2f}% Best Test Macro: {100 * best_test_macro:.2f}%')
        return best_val, best_test_micro, best_test_macro


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    print(args)
    
    set_seed(args.seed)
    
    # Setup
    root = os.path.abspath(os.path.dirname(__file__))
    device = f'cuda:{args.cuda}'
    
    # Create directories
    os.makedirs(f'{root}/teacher', exist_ok=True)
    os.makedirs(f'{root}/syn', exist_ok=True)
    
    # Load data (keep on CPU)
    data = torch.load(f'./data/processed/{args.dataset}.pt')
    labels = torch.LongTensor(data.labels).to(device)
    feats = data.feats  # CPU
    adjs = data.adjs    # CPU edge indices
    
    nclass = int(labels.max() + 1)
    num_steps = feats.shape[0]
    n_nodes = feats.shape[1]
    d = feats.shape[-1]
    
    # Create normalized SparseTensors for full-graph propagation (CPU)
    global_adjs = []
    for edge_index in adjs:
        adj = sp.csr_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(n_nodes, n_nodes)
        )
        adj, _ = utils.to_tensor(adj, feats[0], device='cpu')
        adj = utils.normalize_adj_tensor(adj, sparse=True)
        adj = SparseTensor(
            row=adj._indices()[0], col=adj._indices()[1],
            value=adj._values(), sparse_sizes=adj.size()
        ).t()
        global_adjs.append(adj)
    
    idx_train, idx_val, idx_test = data.train_nodes, data.val_nodes, data.test_nodes
    
    # Load pre-sampled batches
    train_batch = torch.load(f'{root}/data/splited/{args.dataset}_train_batch.pt')
    val_batch = torch.load(f'{root}/data/splited/{args.dataset}_val_batch.pt')
    test_batch = torch.load(f'{root}/data/splited/{args.dataset}_test_batch.pt')
    
    print(f'Dataset: {args.dataset}')
    print(f'Nodes: {n_nodes}, Features: {d}, Classes: {nclass}, Time steps: {num_steps}')
    print(f'Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}')
    
    # Initialize teacher model
    teacher_model = get_model(args.teacher_model, d, args.hidden, nclass, num_steps,
                              args.nlayers, args.dropout, device)
    
    # Initialize condenser
    condenser = LargeCondenser(
        args, feats, adjs, global_adjs, labels,
        idx_train, idx_val, idx_test,
        train_batch, val_batch, test_batch, device
    )
    
    # Train or load teacher
    teacher_path = f'{root}/teacher/{args.teacher_model}_{args.dataset}_{args.dropout}_{args.seed}.pt'
    if not os.path.exists(teacher_path):
        condenser.train_teacher(teacher_model)
    teacher_model.load_state_dict(torch.load(teacher_path))
    
    # Evaluate teacher on sampled batches
    teacher_model_cpu = teacher_model.to('cpu')
    Logits = []
    Labels = []
    for batch in test_batch:
        output = teacher_model_cpu.predict(feats=batch["Feats"], adjs=batch["Adjs"], masks=batch["mask"])
        Logits.append(output)
        Labels.append(labels[batch["nodes"]])
    Logits = torch.cat(Logits, dim=0).argmax(1)
    Labels = torch.cat(Labels, dim=0)
    micro_f1 = f1_score(Logits.cpu(), Labels.cpu(), average='micro')
    macro_f1 = f1_score(Logits.cpu(), Labels.cpu(), average='macro')
    print(f'Teacher Test Micro: {100 * micro_f1:.2f}%  Teacher Test Macro: {100 * macro_f1:.2f}%')
    
    # Condensation
    print("Starting condensation...")
    teacher_model = teacher_model.to(device)
    condenser.condensation(teacher_model)


if __name__ == '__main__':
    main()
