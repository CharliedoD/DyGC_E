import os
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset

from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.utils import to_undirected
from torch_geometric.loader import *
from sklearn.model_selection import train_test_split
from collections import defaultdict, namedtuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.graph_utils import GraphData


def get_dataset(name, normalize_features=True, transform=None, year=2020):
    if name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        root = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    else:
        root = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

    if name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        dataset = PygNodePropPredDataset(name=name, root=root, transform=T.ToSparseTensor())

    if transform is not None and normalize_features:#ogb不管用
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:#归一
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    
    subgraph = filter(dataset)
    return subgraph


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def filter(dataset):
    # Create a mask for nodes within the first 20 years
    split_idx = dataset.get_idx_split() 
    train_nodes = split_idx["train"]
    valid_nodes = split_idx["valid"]
    test_nodes = split_idx["test"]

    data = dataset.data
    n = data.num_nodes

    labels = data.y.squeeze()
    node_year = data.node_year.squeeze()  # Remove extra dimensions if any

    masks_inter = []
    init_mask = node_year <= 2010
    init_mask = torch.nonzero(init_mask).squeeze()
    masks_inter.append(init_mask)
    for year in range(2011, 2020, 2):
        mask = (node_year >= year) & (node_year <= year+1)
        mask = torch.nonzero(mask).squeeze()
        masks_inter.append(mask)

    feats = []
    adjs = []

    feat, idx_train = data.x, train_nodes
    feat_train = feat[idx_train]
    scaler = StandardScaler()
    scaler.fit(feat_train)
    feat = scaler.transform(feat)
    feat = torch.from_numpy(feat).float()

    edge_index = data.edge_index
    edge_index = to_undirected(edge_index=edge_index, edge_attr=None, num_nodes=data.num_nodes)
    edge_index = edge_index[0]

    #for mask_int, mask_his in zip(masks_inter, masks_his):
    for mask_int in masks_inter:
        edge_mask = torch.isin(edge_index[0], mask_int)|torch.isin(edge_index[1], mask_int)
        filtered_edge_index = edge_index[:, edge_mask]
        print(filtered_edge_index.shape)
        adjs.append(filtered_edge_index)
        feats.append(feat)
    feats = torch.stack(feats)
    graph_data = GraphData(feats, labels, adjs, train_nodes, valid_nodes, test_nodes)
    
    # Save to project root data directory
    save_path = os.path.join(os.path.dirname(__file__), '../../data/processed/arxiv.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    graph_data.save(save_path)
    return feats


class Pyg2Dpr(Dataset):#input dataset and get the divided one. if we input partitioned dataset, then we can get what we want
    def __init__(self, pyg_data, **kwargs):

        pyg_data.edge_index = to_undirected(edge_index=pyg_data.edge_index, edge_attr=None, num_nodes=pyg_data.num_nodes)
        pyg_data.edge_index = pyg_data.edge_index[0]
        n = pyg_data.num_nodes

        self.timestep = pyg_data.node_year.numpy()
        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),#化为普通的稀疏矩阵
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()
        if self.labels.shape[-1]==107:
            self.labels = np.argmax(self.labels, -1)
        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape
            
        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            self.idx_train, self.idx_val, self.idx_test = get_train_val_test(nnodes=n, val_size=0.10, test_size=0.30, stratify=self.labels)

        print("train val test:",len(self.idx_train),len(self.idx_val),len(self.idx_test))

   
data = get_dataset('ogbn-arxiv')