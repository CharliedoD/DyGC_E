"""
Dynamic Graph Condensation for Small-Scale Graphs.

This script performs graph condensation on small dynamic graphs that can fit
entirely in GPU memory. For large graphs, use condense_large.py instead.

Key differences from large-scale condensation:
- Full graph is loaded into GPU memory
- Uses standard MMD loss (not efficient version)
- No subgraph sampling needed
"""
import os
import gc
import math
import random
import argparse
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import deeprobust.graph.utils as utils
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce
from sklearn.metrics import f1_score

from src.models.DGNN import GCRN, TGCN, TGCN_L, DySAT, STGCN, ROLAND
from src.models.basicgnn import propagater
from src.models.structure_generation import SNN_generator
from src.utils import gcn_norm, MMDLoss
from src.utils.graph_utils import GraphData  # Required for torch.load


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Graph Condensation (Small Scale)')
    
    # Basic settings
    parser.add_argument('--cuda', type=int, default=0, help='GPU device id')
    parser.add_argument('--dataset', type=str, default='dblp', help='Dataset name')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    
    # Model architecture
    parser.add_argument('--nlayers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--K', type=int, default=2, help='Propagation depth for feature aggregation')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--teacher_model', type=str, default='SGC_MLP', 
                       choices=['GCRN', 'TGCN', 'TGCN_L', 'DySAT', 'STGCN', 'ROLAND'])
    parser.add_argument('--val_model', type=str, default='DGCN',
                       choices=['GCRN', 'TGCN', 'TGCN_L', 'DySAT', 'STGCN', 'ROLAND'])
    
    # Condensation settings
    parser.add_argument('--reduction_rate', type=float, default=0.1, help='Graph reduction rate')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='Learning rate for adjacency')
    parser.add_argument('--lr_feat', type=float, default=0.01, help='Learning rate for features')
    parser.add_argument('--lr_model', type=float, default=0.005, help='Learning rate for model')
    
    # Loss settings
    parser.add_argument('--loss_factor', type=float, default=10, help='MMD loss weight')
    parser.add_argument('--temporal_alpha', type=float, default=0.1, help='Temporal smoothing factor')
    
    # Training loops
    parser.add_argument('--condensing_loop', type=int, default=200, help='Condensation iterations')
    parser.add_argument('--teacher_training_loop', type=int, default=1000, help='Teacher training epochs')
    parser.add_argument('--condensing_val_stage', type=int, default=10, help='Validation frequency')
    parser.add_argument('--student_model_loop', type=int, default=1000, help='Student training epochs')
    parser.add_argument('--student_val_stage', type=int, default=50, help='Student validation frequency')
    
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
    """Generate synthetic labels based on class distribution.
    
    Args:
        labels_train: Training labels.
        num_classes: Number of classes.
        reduction_rate: Reduction rate for condensation.
        
    Returns:
        labels_syn: Synthetic labels.
        num_class_dict: Dictionary mapping class to number of synthetic nodes.
    """
    from collections import Counter
    counter = Counter(labels_train.cpu().numpy())
    num_class_dict = {}
    
    sorted_counter = sorted(counter.items(), key=lambda x: x[1])
    labels_syn = []
    syn_class_indices = {}
    
    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * reduction_rate)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]
    
    return labels_syn, num_class_dict


def gcn_norm_temporal(adjs, nnodes):
    """Apply GCN normalization to temporal adjacency matrices.
    
    Args:
        adjs: Dense adjacency matrices of shape (T, N, N).
        nnodes: Number of nodes.
        
    Returns:
        edge_indexs: List of edge indices.
        edge_weights: List of edge weights.
    """
    edge_indexs = []
    edge_weights = []
    for adj in adjs:
        edge_index = torch.nonzero(adj).T
        edge_weight = adj[edge_index[0], edge_index[1]]
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, nnodes)
        edge_indexs.append(edge_index)
        edge_weights.append(edge_weight)
    return edge_indexs, edge_weights


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
class Condenser:
    """Main class for small-scale dynamic graph condensation."""
    
    def __init__(self, args, feats, adjs, labels, idx_train, idx_val, idx_test, device):
        self.args = args
        self.device = device
        self.root = os.path.abspath(os.path.dirname(__file__))
        
        # Data
        self.feats = feats
        self.adjs = adjs
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        
        # Derived info
        self.num_steps = feats.shape[0]
        self.n_nodes = feats.shape[1]
        self.d = feats.shape[-1]
        self.nclass = int(labels.max() + 1)
        
        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]
        self.feat_train = feats[:, idx_train, :]
        
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
            nfeat=self.d, nnodes=self.nnodes_syn, 
            n_steps=self.num_steps, args=self.args
        ).to(self.device)
        
        print(f'Synthetic graph size: {self.feat_syn.shape}')
        
    def train_teacher(self, teacher_model):
        """Train the teacher model on full data."""
        patience = 150
        stopping_counter = 0
        best_val = 0
        best_test_micro = 0
        best_test_macro = 0
        
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr=self.args.lr_model)
        
        for epoch in tqdm(range(1, self.args.teacher_training_loop + 1), desc="Training Teacher"):
            teacher_model.train()
            optimizer.zero_grad()
            output = teacher_model.forward(self.feats, self.adjs)
            loss = F.nll_loss(output[self.idx_train], self.labels_train)
            loss.backward()
            optimizer.step()
            
            # Evaluation
            output_predict = teacher_model.predict(self.feats, self.adjs)
            y_val_pred = output_predict[self.idx_val].argmax(1)
            y_test_pred = output_predict[self.idx_test].argmax(1)
            
            f1_val_micro = f1_score(self.labels_val.cpu(), y_val_pred.cpu(), average='micro')
            f1_test_micro = f1_score(self.labels_test.cpu(), y_test_pred.cpu(), average='micro')
            f1_test_macro = f1_score(self.labels_test.cpu(), y_test_pred.cpu(), average='macro')
            
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss.item():.4f}, '
                      f'Best Test Micro-F1: {100 * best_test_micro:.2f}%  '
                      f'Best Test Macro-F1: {100 * best_test_macro:.2f}%')
            
            if f1_val_micro > best_val:
                best_val = f1_val_micro
                best_test_micro = f1_test_micro
                best_test_macro = f1_test_macro
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
    
    def condensation(self, teacher_model, val_model, optimizer_val):
        """Main condensation loop."""
        start = time.perf_counter()
        
        optimizer_feat = optim.Adam([self.feat_syn], lr=self.args.lr_feat)
        optimizer_pge = optim.Adam(self.structure_generator.parameters(), lr=self.args.lr_adj)
        propagate = propagater().to(self.device)
        
        # ===== Precompute aggregated features for original data =====
        feats_org, coeff, coeff_sum = self._precompute_features(propagate)
        
        # ===== Condensation loop =====
        best_val = 0
        best_test_micro = 0
        best_test_macro = 0
        loss_fn_MMD = MMDLoss()
        
        for i in tqdm(range(1, self.args.condensing_loop + 1), desc="Condensing"):
            # Generate synthetic adjacency
            adjs_syn = self.structure_generator(self.feat_syn)
            edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, self.nnodes_syn)
            
            # Compute aggregated features for synthetic data
            FEAT_syn = self._compute_syn_features(propagate, edge_indexs_syn, edge_weights_syn)
            
            # Split by class
            feats_syn = []
            for c in range(self.nclass):
                if c in self.num_class_dict:
                    index = torch.where(self.labels_syn == c)[0]
                    feats_syn.append(FEAT_syn[index])
            
            # Compute MMD loss
            mmd_loss = torch.tensor(0.0).to(self.device)
            c_idx = 0
            for c in range(self.nclass):
                if c in self.num_class_dict:
                    mmd_loss += coeff[c_idx] * loss_fn_MMD(feats_org[c_idx], feats_syn[c_idx])
                    c_idx += 1
            mmd_loss = mmd_loss / coeff_sum
            
            # Teacher loss
            teacher_model.train()
            output_syn = teacher_model.forward(self.feat_syn, edge_indexs_syn, edge_weights_syn)
            hard_loss = F.nll_loss(output_syn, self.labels_syn)
            
            # Total loss
            loss = hard_loss + self.args.loss_factor * mmd_loss
            
            # Update
            optimizer_pge.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()
            optimizer_pge.step()
            optimizer_feat.step()
            
            # Validation
            if i >= 100 and i % self.args.condensing_val_stage == 0:
                best_val, best_test_micro, best_test_macro = self._validate(
                    teacher_model, val_model, optimizer_val,
                    best_val, best_test_micro, best_test_macro, i
                )
        
        end = time.perf_counter()
        print(f'Condensation Duration: {round(end - start)}s')
    
    def _precompute_features(self, propagate):
        """Precompute aggregated features for original training data."""
        alpha = self.args.temporal_alpha
        concat_feat_all = []
        agg_memory = []
        
        for time_step in range(self.num_steps):
            concat_feat = self.feat_train[time_step]
            temp = self.feats[time_step].to(self.device)
            adj = self.adjs[time_step]
            
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
        FEAT_org = concat_feat_all.permute(1, 0, 2)  # (N_train, T, D')
        
        # Split by class
        feats_org = []
        coeff = []
        coeff_sum = 0
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index = torch.where(self.labels_train == c)[0].to('cpu')
                coe = self.num_class_dict[c] / max(self.num_class_dict.values())
                coeff_sum += coe
                coeff.append(coe)
                feat_org_c = FEAT_org[index]
                feats_org.append(feat_org_c)
        
        coeff_sum = torch.tensor(coeff_sum).to(self.device)
        return feats_org, coeff, coeff_sum
    
    def _compute_syn_features(self, propagate, edge_indexs_syn, edge_weights_syn):
        """Compute aggregated features for synthetic data."""
        alpha = self.args.temporal_alpha
        concat_feat_syn_all = []
        aggr_syn_memory = []
        
        for time_step in range(self.num_steps):
            concat_feat_syn = self.feat_syn[time_step]
            temp = self.feat_syn[time_step]
            edge_index_syn = edge_indexs_syn[time_step]
            edge_weight_syn = edge_weights_syn[time_step]
            
            for k in range(self.args.K):
                aggr_syn = propagate(edge_index=edge_index_syn, x=temp, edge_weight=edge_weight_syn)
                if time_step == 0:
                    aggr_syn_memory.append(aggr_syn)
                else:
                    aggr_syn = alpha * aggr_syn_memory[k] + (1 - alpha) * aggr_syn
                    aggr_syn_memory[k] = aggr_syn
                concat_feat_syn = torch.cat((concat_feat_syn, aggr_syn), dim=1)
                temp = aggr_syn
            concat_feat_syn_all.append(concat_feat_syn)
        
        concat_feat_syn_all = torch.stack(concat_feat_syn_all)
        FEAT_syn = concat_feat_syn_all.permute(1, 0, 2)  # (N_syn, T, D')
        return FEAT_syn
    
    def _validate(self, teacher_model, val_model, optimizer, 
                  best_val, best_test_micro, best_test_macro, epoch):
        """Validate synthetic data quality."""
        adjs_syn = self.structure_generator(self.feat_syn, inference=True)
        edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, self.nnodes_syn)
        
        val_model.initialize()
        teacher_output_syn = teacher_model.predict(self.feat_syn, edge_indexs_syn, edge_weights_syn)
        acc = utils.accuracy(teacher_output_syn, self.labels_syn)
        print(f'Epoch {epoch} Teacher on syn accuracy= {100*acc.item():.2f}')
        
        test_micros, test_macros = [], []
        
        for j in range(self.args.student_model_loop):
            val_model.train()
            optimizer.zero_grad()
            output_syn = val_model.forward(self.feat_syn, edge_indexs_syn, edge_weights_syn)
            loss = F.nll_loss(output_syn, self.labels_syn)
            loss.backward()
            optimizer.step()
            
            if j % self.args.student_val_stage == 0:
                output = val_model.predict(self.feats.to(self.device), self.adjs)
                y_val_pred = output[self.idx_val].argmax(1)
                y_test_pred = output[self.idx_test].argmax(1)
                
                f1_val_micro = f1_score(self.labels_val.cpu(), y_val_pred.cpu(), average='micro')
                f1_test_micro = f1_score(self.labels_test.cpu(), y_test_pred.cpu(), average='micro')
                f1_test_macro = f1_score(self.labels_test.cpu(), y_test_pred.cpu(), average='macro')
                
                test_micros.append(f1_test_micro)
                test_macros.append(f1_test_macro)
                
                if f1_val_micro >= best_val:
                    best_val = f1_val_micro
                    best_test_micro = f1_test_micro
                    best_test_macro = f1_test_macro
                    torch.save(self.feat_syn, f'{self.root}/syn/feat_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}.pt')
                    torch.save(adjs_syn, f'{self.root}/syn/adj_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}.pt')
                    torch.save(self.labels_syn, f'{self.root}/syn/label_{self.args.dataset}_{self.args.reduction_rate}_{self.args.seed}.pt')
        
        print("test Micro:", ', '.join([f"{t*100:.2f}" for t in test_micros]))
        print("test Macro:", ', '.join([f"{t*100:.2f}" for t in test_macros]))
        print(f'Epoch {epoch} | Best test Micro F1: {100 * best_test_micro:.2f}%  Best test Macro F1: {100 * best_test_macro:.2f}%')
        
        return best_val, best_test_micro, best_test_macro


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    print(args)
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    set_seed(args.seed)
    
    # Setup
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    device = f'cuda:{args.cuda}'
    
    # Create directories
    os.makedirs(f'{root}/teacher', exist_ok=True)
    os.makedirs(f'{root}/syn', exist_ok=True)
    
    # Load data
    data = torch.load(f'./data/processed/{args.dataset}.pt')
    feats = data.feats.to(device)
    labels = torch.LongTensor(data.labels).to(device)
    n_nodes = feats.shape[1]
    nclass = int(labels.max() + 1)
    num_steps = feats.shape[0]
    d = feats.shape[-1]
    
    # Normalize adjacency matrices
    adjs = []
    for edge_index in data.adjs:
        edge_weight = torch.ones(edge_index.size(1))
        adj = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=(n_nodes, n_nodes))
        adj = utils.normalize_adj_tensor(adj, sparse=True)
        adj = SparseTensor(
            row=adj._indices()[0], col=adj._indices()[1],
            value=adj._values(), sparse_sizes=adj.size()
        ).t()
        adjs.append(adj.to(device))
    
    idx_train, idx_val, idx_test = data.train_nodes, data.val_nodes, data.test_nodes
    
    print(f'Dataset: {args.dataset}')
    print(f'Nodes: {n_nodes}, Features: {d}, Classes: {nclass}, Time steps: {num_steps}')
    print(f'Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}')
    
    # Initialize models
    teacher_model = get_model(args.teacher_model, d, args.hidden, nclass, num_steps, 
                              args.nlayers, args.dropout, device)
    val_model = get_model(args.val_model, d, args.hidden, nclass, num_steps,
                          args.nlayers, args.dropout, device)
    optimizer = optim.Adam(val_model.parameters(), lr=args.lr_model)
    
    # Train or load teacher
    teacher_path = f'{root}/teacher/{args.teacher_model}_{args.dataset}_{args.dropout}_{args.seed}.pt'
    
    # Initialize condenser
    condenser = Condenser(args, feats, adjs, labels, idx_train, idx_val, idx_test, device)
    
    if not os.path.exists(teacher_path):
        condenser.train_teacher(teacher_model)
    teacher_model.load_state_dict(torch.load(teacher_path))
    teacher_model.eval()
    
    # Evaluate teacher
    output = teacher_model.predict(feats, adjs)
    y_test_pred = output[idx_test].max(dim=1)[1]
    labels_test = labels[idx_test]
    f1_test_micro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='micro')
    f1_test_macro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='macro')
    print(f'Teacher Test Micro: {100 * f1_test_micro:.2f}%  Test Macro: {100 * f1_test_macro:.2f}%')
    
    # Condensation
    print("Starting condensation...")
    condenser.condensation(teacher_model, val_model, optimizer)


if __name__ == '__main__':
    main()
