"""
Data preprocessing script for dynamic graphs.

Converts raw .npz data files to processed .pt files for use in training.
Run from project root: python data/scripts/get_data.py --data dblp
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.graph_utils import GraphData


def adjacency_to_edge_index(adj):
    """Convert adjacency matrix to edge index format."""
    edge_index = np.nonzero(adj)
    edge_index = np.array(edge_index)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index


def main():
    parser = argparse.ArgumentParser(description='Preprocess dynamic graph data')
    parser.add_argument('--data', type=str, default='dblp', help='Dataset name')
    parser.add_argument('--normalize', action='store_true', help='Normalize features')
    args = parser.parse_args()
    
    name = args.data
    print(f"Processing dataset: {name}")
    
    # Load raw data (path relative to project root)
    raw_path = f'data/raw/{name}.npz'
    if not os.path.exists(raw_path):
        print(f"Error: Raw data file not found at {raw_path}")
        return
    
    file = np.load(raw_path)
    
    feats = file['attmats']  # (N, T, D) node features
    labels = file['labels']  # (N, C) node labels
    adjs = file['adjs']      # (T, N, N) graph snapshots
    
    print(f"Features shape: {feats.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Adjacencies shape: {adjs.shape}")
    
    # Convert to PyTorch tensors
    feats = torch.tensor(feats, dtype=torch.float32).transpose(0, 1)  # (T, N, D)
    
    # Optionally normalize features
    if args.normalize or name == 'reddit':
        feats = F.normalize(feats, p=2, dim=2)
        print("Features normalized")
    
    # Convert labels to class indices
    labels = torch.tensor(labels, dtype=torch.float32)
    labels = torch.argmax(labels, dim=1)
    
    # Convert adjacencies to edge indices
    edge_indexs = []
    for adj in adjs:
        edge_index = adjacency_to_edge_index(adj)
        print(f"Edge index shape: {edge_index.shape}")
        edge_indexs.append(edge_index)
    
    # Split nodes into train/val/test
    train_nodes, test_nodes = train_test_split(
        torch.arange(labels.size(0)),
        train_size=0.5,
        test_size=0.5,
        random_state=42,
        stratify=labels
    )
    
    val_nodes, test_nodes = train_test_split(
        test_nodes,
        train_size=0.2 / 0.5,
        random_state=42,
        stratify=labels[test_nodes]
    )
    
    print(f"Train nodes: {len(train_nodes)}")
    print(f"Val nodes: {len(val_nodes)}")
    print(f"Test nodes: {len(test_nodes)}")
    
    # Create and save GraphData
    graph_data = GraphData(
        feats=feats,
        labels=labels.numpy(),
        adjs=edge_indexs,
        train_nodes=train_nodes,
        val_nodes=val_nodes,
        test_nodes=test_nodes
    )
    
    # Ensure processed directory exists
    os.makedirs('data/processed', exist_ok=True)
    graph_data.save(f'data/processed/{name}.pt')


if __name__ == '__main__':
    main()
