"""
Download and preprocess OGB arxiv dataset for dynamic graph learning.

Run from project root: python data/scripts/get_arxiv.py
"""
import os
import sys
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.graph_utils import GraphData


def get_dataset(name, normalize_features=True, transform=None):
    """Download and load OGB dataset."""
    # Store dataset in data/raw directory
    root = os.path.join(os.path.dirname(__file__), '../raw')
    os.makedirs(root, exist_ok=True)
    
    dataset = PygNodePropPredDataset(name=name, root=root, transform=T.ToSparseTensor())
    
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    
    return dataset


def process_arxiv(dataset):
    """Process arxiv dataset into temporal snapshots."""
    split_idx = dataset.get_idx_split() 
    train_nodes = split_idx["train"]
    valid_nodes = split_idx["valid"]
    test_nodes = split_idx["test"]

    data = dataset.data
    labels = data.y.squeeze()
    node_year = data.node_year.squeeze()

    # Create temporal masks (papers grouped by year ranges)
    masks_inter = []
    init_mask = node_year <= 2010
    init_mask = torch.nonzero(init_mask).squeeze()
    masks_inter.append(init_mask)
    
    for year in range(2011, 2020, 2):
        mask = (node_year >= year) & (node_year <= year + 1)
        mask = torch.nonzero(mask).squeeze()
        masks_inter.append(mask)

    # Normalize features using training set statistics
    feat, idx_train = data.x, train_nodes
    feat_train = feat[idx_train]
    scaler = StandardScaler()
    scaler.fit(feat_train)
    feat = scaler.transform(feat)
    feat = torch.from_numpy(feat).float()

    # Get undirected edge index
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index=edge_index, edge_attr=None, num_nodes=data.num_nodes)
    edge_index = edge_index[0]

    # Create temporal snapshots
    feats = []
    adjs = []
    
    for mask_int in masks_inter:
        edge_mask = torch.isin(edge_index[0], mask_int) | torch.isin(edge_index[1], mask_int)
        filtered_edge_index = edge_index[:, edge_mask]
        print(f"Timestep edges: {filtered_edge_index.shape}")
        adjs.append(filtered_edge_index)
        feats.append(feat)
    
    feats = torch.stack(feats)
    
    # Create and save GraphData
    graph_data = GraphData(feats, labels, adjs, train_nodes, valid_nodes, test_nodes)
    
    # Save to processed directory
    save_path = 'data/processed/arxiv.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    graph_data.save(save_path)
    
    return graph_data


def main():
    print("Downloading and processing ogbn-arxiv dataset...")
    dataset = get_dataset('ogbn-arxiv')
    graph_data = process_arxiv(dataset)
    print("Done!")


if __name__ == '__main__':
    main()
