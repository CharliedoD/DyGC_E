"""
Test script for evaluating condensed small-scale dynamic graphs.

This script trains a student model on the condensed graph and evaluates
it on the original test set.
"""
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

warnings.simplefilter("ignore", category=FutureWarning)


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Test Condensed Graph (Small Scale)')
    
    # Basic settings
    parser.add_argument('--cuda', type=int, default=0, help='GPU device id')
    parser.add_argument('--dataset', type=str, default='dblp', help='Dataset name')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for condensation')
    
    # Model architecture
    parser.add_argument('--nlayers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    
    # Test settings
    parser.add_argument('--test_model', type=str, default='TGCN',
                       choices=['GCRN', 'TGCN', 'TGCN_L', 'DySAT', 'STGCN', 'ROLAND'])
    parser.add_argument('--reduction_rate', type=float, default=0.1, help='Reduction rate used in condensation')
    parser.add_argument('--lr_model', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--test_loop', type=int, default=1000, help='Training epochs')
    parser.add_argument('--val_stage', type=int, default=50, help='Validation frequency')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
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
                              num_timesteps=num_steps, nlayers=nlayers, nconv=3, dropout=dropout),
        'ROLAND': lambda: ROLAND(in_features=d, hidden_feature=hidden, out_features=nclass,
                                num_timesteps=num_steps, nlayers=nlayers, dropout=dropout, update='GRU'),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    return model_map[model_name]().to(device)


def main():
    args = parse_args()
    print(args)
    
    set_seed(args.seed)
    
    # Setup
    root = os.path.abspath(os.path.dirname(__file__))
    device = f'cuda:{args.cuda}'
    
    # Load original data for evaluation
    data = torch.load(f'./data/processed/{args.dataset}.pt')
    feats = data.feats.to(device)
    labels = torch.LongTensor(data.labels).to(device)
    n_nodes = feats.shape[1]
    
    if args.dataset == 'reddit':
        feats = F.normalize(feats, p=2, dim=2)
    
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
    labels_train, labels_val, labels_test = labels[idx_train], labels[idx_val], labels[idx_test]
    
    nclass = int(labels.max() + 1)
    num_steps = feats.shape[0]
    d = feats.shape[-1]
    
    print(f'Dataset: {args.dataset}')
    print(f'Nodes: {n_nodes}, Features: {d}, Classes: {nclass}, Time steps: {num_steps}')
    
    # Load condensed data
    feat_syn = torch.load(f'{root}/syn/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    adjs_syn = torch.load(f'{root}/syn/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    labels_syn = torch.load(f'{root}/syn/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    nnodes_syn = len(labels_syn)
    
    print(f'Condensed graph: {nnodes_syn} nodes (reduction rate: {args.reduction_rate})')
    
    # Initialize test model
    test_model = get_model(args.test_model, d, args.hidden, nclass, num_steps,
                           args.nlayers, args.dropout, device)
    optimizer = optim.Adam(test_model.parameters(), lr=args.lr_model)
    
    # Normalize condensed adjacency
    edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, nnodes_syn)
    
    # Training loop
    best_val = 0
    best_test_micro = 0
    best_test_macro = 0
    
    print("Training on condensed graph...")
    for j in tqdm(range(1, args.test_loop + 1), desc="Testing"):
        test_model.train()
        optimizer.zero_grad()
        output_syn = test_model.forward(feat_syn, edge_indexs_syn, edge_weights_syn)
        loss = F.nll_loss(output_syn, labels_syn)
        loss.backward()
        optimizer.step()
        
        if j % args.val_stage == 0:
            output = test_model.predict(feats, adjs)
            y_val_pred = output[idx_val].argmax(1)
            y_test_pred = output[idx_test].argmax(1)
            
            f1_val_micro = f1_score(labels_val.cpu(), y_val_pred.cpu(), average='micro')
            f1_test_micro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='micro')
            f1_test_macro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='macro')
            
            if f1_val_micro >= best_val:
                best_val = f1_val_micro
                best_test_micro = f1_test_micro
                best_test_macro = f1_test_macro
            
            print(f"loop {j}: Test micro: {f1_test_micro*100:.2f} Test macro: {f1_test_macro*100:.2f}")
    
    print(f"\nBest Test Micro: {best_test_micro*100:.2f} Best Test Macro: {best_test_macro*100:.2f}")


if __name__ == '__main__':
    main()
