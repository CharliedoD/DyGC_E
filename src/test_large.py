"""
Test script for evaluating condensed large-scale dynamic graphs.

This script trains a student model on the condensed graph and evaluates
it on the original test set using sampled batches.
"""
import os
import gc
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import deeprobust.graph.utils as utils
from torch_sparse import SparseTensor
from sklearn.metrics import f1_score

from src.models.DGNN import GCRN, TGCN, TGCN_L, DySAT, STGCN, ROLAND
from src.utils import gcn_norm
from src.utils.graph_utils import GraphData  # Required for torch.load

warnings.filterwarnings("ignore")


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Test Condensed Graph (Large Scale)')
    
    # Basic settings
    parser.add_argument('--cuda', type=int, default=0, help='GPU device id')
    parser.add_argument('--dataset', type=str, default='arxiv', help='Dataset name')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for condensation')
    parser.add_argument('--testseed', type=int, default=2025, help='Random seed for testing')
    
    # Model architecture
    parser.add_argument('--nlayers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    
    # Test settings
    parser.add_argument('--test_model', type=str, default='TGCN_L',
                       choices=['GCRN', 'TGCN', 'TGCN_L', 'DySAT', 'STGCN', 'ROLAND'])
    parser.add_argument('--reduction_rate', type=float, default=0.1, help='Reduction rate used in condensation')
    parser.add_argument('--lr_model', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_loop', type=int, default=2000, help='Training epochs')
    parser.add_argument('--val_stage', type=int, default=100, help='Validation frequency')
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size for evaluation')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def main():
    args = parse_args()
    print(args)
    
    set_seed(args.testseed)
    
    # Setup
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    device = f'cuda:{args.cuda}'
    
    # Load original data info
    data = torch.load(f'./data/processed/{args.dataset}.pt')
    feats = data.feats  # CPU
    adjs = data.adjs    # CPU
    labels = torch.LongTensor(data.labels).to(device)
    
    idx_train, idx_val, idx_test = data.train_nodes, data.val_nodes, data.test_nodes
    labels_train, labels_val, labels_test = labels[idx_train], labels[idx_val], labels[idx_test]
    
    num_steps = feats.shape[0]
    n_nodes = feats.shape[1]
    d = feats.shape[2]
    nclass = int(labels.max() + 1)
    
    print(f'Dataset: {args.dataset}')
    print(f'Nodes: {n_nodes}, Features: {d}, Classes: {nclass}, Time steps: {num_steps}')
    
    # Load sampled batches for evaluation
    train_batch = torch.load(f'{root}/data/splited/{args.dataset}_train_batch.pt')
    val_batch = torch.load(f'{root}/data/splited/{args.dataset}_val_batch.pt')
    test_batch = torch.load(f'{root}/data/splited/{args.dataset}_test_batch.pt')
    
    # Load condensed data
    feat_syn = torch.load(f'{root}/syn/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    adjs_syn = torch.load(f'{root}/syn/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    labels_syn = torch.load(f'{root}/syn/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    nnodes_syn = len(labels_syn)
    
    print(f'Condensed graph: {nnodes_syn} nodes (reduction rate: {args.reduction_rate})')
    
    # Initialize test model
    test_model = get_model(args.test_model, d, args.hidden, nclass, num_steps,
                           args.nlayers, args.dropout, device)
    optimizer = optim.AdamW(test_model.parameters(), lr=args.lr_model)
    test_model.initialize()
    
    # Normalize condensed adjacency
    edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, nnodes_syn)
    
    # Training loop
    best_val = 0
    best_test_micro = 0
    best_test_macro = 0
    
    print("Training on condensed graph...")
    for epoch in range(1, args.test_loop + 1):
        test_model.train()
        output_syn = test_model.forward(feat_syn, edge_indexs_syn, edge_weights_syn)
        loss = F.nll_loss(output_syn, labels_syn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % args.val_stage == 0:
            val_micro, val_macro = test_on_batches(val_batch, test_model, labels, device)
            test_micro, test_macro = test_on_batches(test_batch, test_model, labels, device)
            
            print(f'Epoch: {epoch:02d}, Loss: {loss.item():.4f}, '
                  f'Test Micro: {100 * test_micro:.2f}% Test Macro: {100 * test_macro:.2f}%')
            
            if val_micro > best_val:
                best_val = val_micro
                best_test_micro = test_micro
                best_test_macro = test_macro
    
    print(f'\nBest Test Micro: {100 * best_test_micro:.2f}% Best Test Macro: {100 * best_test_macro:.2f}%')


if __name__ == '__main__':
    main()
