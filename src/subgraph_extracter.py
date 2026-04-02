"""
Subgraph Extraction for Large-Scale Dynamic Graphs.

This script preprocesses large dynamic graphs by extracting sampled subgraphs
for efficient training and evaluation. Must be run before condense_large.py.

Two sampling modes are supported:
1. Independent sampling: Each time step samples neighbors independently
2. Shared sampling: All time steps share the same sampled node set

Usage:
    python subgraph_extracter.py --dataset arxiv --graph_size 100000
"""
import os
import sys
import random
import argparse
import numpy as np
import warnings

import torch
import scipy.sparse as sp
import deeprobust.graph.utils as utils
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from src.utils.graph_utils import GraphData  # Required for torch.load

warnings.filterwarnings("ignore")


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Subgraph Extraction for Large Graphs')
    
    parser.add_argument('--dataset', type=str, default='arxiv', help='Dataset name')
    parser.add_argument('--cuda', type=int, default=0, help='GPU device id')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--share', type=int, default=0, 
                       help='Sampling method: 0=independent, 1=shared')
    parser.add_argument('--sample_depth', type=int, default=3,
                       help='Depth of neighbor sampling')
    parser.add_argument('--graph_size', type=int, default=100000,
                       help='Maximum number of nodes per batch')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ============================================================================
# Sampling Functions
# ============================================================================
class TemporalSubgraphExtractor:
    """Extract temporal subgraphs from large dynamic graphs.
    
    Args:
        feats: Node features of shape (T, N, D).
        adjs: List of edge indices for each time step.
        sample_depth: Depth of neighbor sampling.
    """
    
    def __init__(self, feats, adjs, sample_depth=3):
        self.feats = feats
        self.adjs = adjs
        self.sample_depth = sample_depth
        self.num_timesteps = feats.shape[0]
    
    def sample_independent(self, nodes):
        """Sample independently for each time step.
        
        Each time step samples its own neighbor set. The returned mask
        indicates positions of seed nodes in the sampled subgraph.
        
        Args:
            nodes: Seed node indices.
            
        Returns:
            Dictionary containing Feats, Adjs, nodes, and mask.
        """
        Feats = []
        Adjs = []
        
        for x, adj in zip(self.feats, self.adjs):
            graph = Data(x=x, edge_index=adj)
            loader = NeighborLoader(
                graph,
                input_nodes=nodes,
                num_neighbors=[-1] * self.sample_depth,
                batch_size=nodes.shape[0]
            )
            sampled_data = next(iter(loader))
            
            feat = sampled_data.x
            edge_index = sampled_data.edge_index
            
            # Convert to normalized SparseTensor
            adj_sparse = sp.csr_matrix(
                (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                shape=(feat.shape[0], feat.shape[0])
            )
            adj_tensor, feat = utils.to_tensor(adj_sparse, feat, device='cpu')
            if utils.is_sparse_tensor(adj_tensor):
                adj_tensor = utils.normalize_adj_tensor(adj_tensor, sparse=True)
            else:
                adj_tensor = utils.normalize_adj_tensor(adj_tensor)
            adj_sparse_tensor = SparseTensor(
                row=adj_tensor._indices()[0],
                col=adj_tensor._indices()[1],
                value=adj_tensor._values(),
                sparse_sizes=adj_tensor.size()
            ).t()
            
            Adjs.append(adj_sparse_tensor)
            Feats.append(feat)
            mask = sampled_data.input_id
        
        return {
            "nodes": nodes,
            "Feats": Feats,
            "Adjs": Adjs,
            "mask": mask
        }
    
    def sample_shared(self, nodes):
        """Sample with shared node set across all time steps.
        
        First collects all sampled nodes from all time steps, then extracts
        subgraphs using the combined node set.
        
        Args:
            nodes: Seed node indices.
            
        Returns:
            Dictionary containing Feats, Adjs, nodes, and mask.
        """
        # First pass: collect all sampled nodes
        all_sampled_nodes = set(nodes.tolist())
        for x, adj in zip(self.feats, self.adjs):
            graph = Data(x=x, edge_index=adj)
            loader = NeighborLoader(
                graph,
                input_nodes=nodes,
                num_neighbors=[-1] * self.sample_depth,
                batch_size=nodes.shape[0]
            )
            sampled_graph = next(iter(loader))
            sampled_nodes = sampled_graph.n_id
            all_sampled_nodes.update(sampled_nodes.tolist())
        
        # Build node mapping
        all_sampled_nodes = np.array(list(all_sampled_nodes))
        value_to_index = {value: idx for idx, value in enumerate(all_sampled_nodes)}
        mask = [value_to_index[x] for x in np.array(nodes)]
        
        all_sampled_nodes = torch.tensor(all_sampled_nodes)
        mask = torch.tensor(mask)
        
        # Second pass: extract subgraph
        Feats = []
        Adjs = []
        for x, adj in zip(self.feats, self.adjs):
            graph = Data(x=x, edge_index=adj)
            sub_graph = graph.subgraph(all_sampled_nodes)
            feat = sub_graph.x
            edge_index = sub_graph.edge_index
            
            adj_sparse = sp.csr_matrix(
                (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                shape=(feat.shape[0], feat.shape[0])
            )
            adj_tensor, feat = utils.to_tensor(adj_sparse, feat, device='cpu')
            adj_tensor = utils.normalize_adj_tensor(adj_tensor, sparse=True)
            adj_sparse_tensor = SparseTensor(
                row=adj_tensor._indices()[0],
                col=adj_tensor._indices()[1],
                value=adj_tensor._values(),
                sparse_sizes=adj_tensor.size()
            ).t()
            
            Adjs.append(adj_sparse_tensor)
            Feats.append(feat)
        
        return {
            "nodes": nodes,
            "Feats": Feats,
            "Adjs": Adjs,
            "mask": mask
        }
    
    def sample(self, nodes, share=False):
        """Sample subgraph for given seed nodes.
        
        Args:
            nodes: Seed node indices.
            share: Whether to use shared sampling across time steps.
            
        Returns:
            Dictionary containing Feats, Adjs, nodes, and mask.
        """
        if share:
            return self.sample_shared(nodes)
        else:
            return self.sample_independent(nodes)


def create_batches(extractor, node_loader, share=False):
    """Create batches of sampled subgraphs.
    
    Args:
        extractor: TemporalSubgraphExtractor instance.
        node_loader: DataLoader for node indices.
        share: Whether to use shared sampling.
        
    Returns:
        List of batch dictionaries.
    """
    batches = []
    for nodes in node_loader:
        nodes = torch.tensor(nodes) if not isinstance(nodes, torch.Tensor) else nodes
        batch = extractor.sample(nodes, share=share)
        batches.append(batch)
    return batches


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
    
    # Create output directory
    os.makedirs(f'{root}/data/splited', exist_ok=True)
    
    # Load data
    data = torch.load(f'./data/processed/{args.dataset}.pt')
    feats = data.feats
    adjs = data.adjs
    labels = data.labels
    
    print(f'Dataset: {args.dataset}')
    print(f'Features shape: {feats.shape}')
    print(f'Number of time steps: {feats.shape[0]}')
    print(f'Number of nodes: {feats.shape[1]}')
    
    idx_train, idx_val, idx_test = data.train_nodes, data.val_nodes, data.test_nodes
    
    print(f'Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}')
    
    # Create data loaders
    train_loader = DataLoader(idx_train.tolist(), pin_memory=False, batch_size=args.graph_size)
    val_loader = DataLoader(idx_val.tolist(), pin_memory=False, batch_size=args.graph_size)
    test_loader = DataLoader(idx_test.tolist(), pin_memory=False, batch_size=args.graph_size)
    
    # Initialize extractor
    extractor = TemporalSubgraphExtractor(feats, adjs, sample_depth=args.sample_depth)
    
    # Extract subgraphs
    share = args.share == 1
    print(f"\nExtracting subgraphs (share={share})...")
    
    print("Processing train set...")
    train_batch = create_batches(extractor, train_loader, share=share)
    
    print("Processing val set...")
    val_batch = create_batches(extractor, val_loader, share=share)
    
    print("Processing test set...")
    test_batch = create_batches(extractor, test_loader, share=share)
    
    # Save batches
    suffix = '_share' if share else ''
    torch.save(train_batch, f'{root}/data/splited/{args.dataset}_train_batch{suffix}.pt')
    torch.save(val_batch, f'{root}/data/splited/{args.dataset}_val_batch{suffix}.pt')
    torch.save(test_batch, f'{root}/data/splited/{args.dataset}_test_batch{suffix}.pt')
    
    print(f"\nSaved batches to {root}/data/splited/")
    print(f"  Train batches: {len(train_batch)}")
    print(f"  Val batches: {len(val_batch)}")
    print(f"  Test batches: {len(test_batch)}")


if __name__ == '__main__':
    main()
