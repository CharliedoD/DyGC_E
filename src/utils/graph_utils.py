"""
Graph utility functions.
"""
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple, Union
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    """Compute GCN-style normalization for edges.
    
    Args:
        edge_index: Edge indices or SparseTensor.
        edge_weight: Edge weights.
        num_nodes: Number of nodes.
        improved: Use improved normalization.
        add_self_loops: Whether to add self-loops.
        flow: Message flow direction.
        dtype: Data type.
        
    Returns:
        Normalized edge_index and edge_weight (or normalized SparseTensor).
    """
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_cos_sim(feature1: Tensor, feature2: Tensor) -> float:
    """Compute cosine similarity between two features."""
    num = torch.dot(feature1, feature2)
    denom = torch.linalg.norm(feature1) * torch.linalg.norm(feature2)
    return num / denom if denom != 0 else 0


def mask_to_index(index, size: int) -> np.ndarray:
    """Convert mask to indices."""
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size: int) -> Tensor:
    """Convert indices to mask."""
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


class GraphData:
    """Container for dynamic graph data.
    
    Attributes:
        feats: Node features of shape (T, N, D).
        labels: Node labels of shape (N,).
        adjs: List of edge indices for each time step.
        train_nodes: Training node indices.
        val_nodes: Validation node indices.
        test_nodes: Test node indices.
    """
    def __init__(self, feats, labels, adjs, train_nodes, val_nodes, test_nodes):
        self.feats = feats
        self.labels = labels
        self.adjs = adjs
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
