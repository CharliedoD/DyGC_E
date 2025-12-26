import os.path as osp
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import deeprobust.graph.utils as utils

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.amazon_products import AmazonProducts
from torch_geometric.datasets.reddit import Reddit
from torch_geometric.datasets.reddit2 import Reddit2
from torch_geometric.datasets.flickr import Flickr
from torch_geometric.datasets.s3dis import S3DIS

from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected, dropout_adj
from torch_geometric.loader import *
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

import math
from collections import defaultdict, namedtuple
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.sparse as ts
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import LSTM
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

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
    

def get_cos_sim(feature1,feature2):
    num = torch.dot(feature1, feature2)  
    denom = torch.linalg.norm(feature1) * torch.linalg.norm(feature2)  
    return num / denom if denom != 0 else 0


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

class GraphData:
    def __init__(self, feats, labels, adjs, train_nodes, val_nodes, test_nodes):
        self.feats = feats
        self.labels = labels
        self.adjs = adjs
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes




import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import MultivariateNormal, kl_divergence
import numpy as np  

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth


    def forward(self, X):

        flat_X = X.view(X.shape[0], -1)
        L2_distances_X = torch.cdist(flat_X, flat_X) ** 2
        K_X = -L2_distances_X[None, ...] / (self.get_bandwidth(L2_distances_X) * self.bandwidth_multipliers.to(L2_distances_X.device)[:, None, None])
        K = torch.exp(K_X).sum(dim=0)

        return K


class RBF_eff(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)

    def forward(self, bandwidth, X, Y):
        
        flat_X = X.view(X.shape[0], -1)
        flat_Y = Y.view(Y.shape[0], -1)

        L2_distances_F = torch.cdist(flat_X, flat_Y) ** 2

        K = -L2_distances_F[None, ...] / (bandwidth * self.bandwidth_multipliers.to(L2_distances_F.device)[:, None, None])

        K = torch.exp(K).sum(dim=0)

        return K
    
class PoliKernel(nn.Module):

    def __init__(self, constant_term=1, degree=2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree


    def forward(self, X):
        X = X.view(X.shape[0], -1)
        K = (torch.matmul(X, X.t()) + self.constant_term) ** self.degree
        return K

class LinearKernel(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, X):
        X = X.view(X.shape[0], -1)
        K = torch.matmul(X, X.t())
        return K

class LaplaceKernel(nn.Module):

    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()


    def forward(self, X):
        X = X.view(X.shape[0], -1)
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] * (self.gammas)[:, None, None]).sum(dim=0)



class MMDLoss_eff(nn.Module):

    def __init__(self):
        super().__init__()
        self.kernel = RBF_eff()
        self.computed_XX = []

    def forward(self, XX, bandwidth, X, Y):
        XY = self.kernel(bandwidth, X, Y).mean()
        YY = self.kernel(bandwidth, Y, Y).mean()
        MMD = XX - 2 * XY + YY
        return MMD
    
    
class MMDLoss(nn.Module):
    def __init__(self, kernel_type='RBF'):
        super().__init__()
        if kernel_type=='RBF':
            self.kernel = RBF()
        elif kernel_type=='Lin':
            self.kernel = LinearKernel()
        elif kernel_type=='Poly':
            self.kernel = PoliKernel()
        elif kernel_type=='Lap':
            self.kernel = LaplaceKernel()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
