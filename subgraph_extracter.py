from models.DGNN import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import argparse
import time
from tqdm import tqdm
import scipy.sparse as sp
import deeprobust.graph.utils as utils
import os 
from torch_sparse import SparseTensor
import random
import numpy as np
from utils import *
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='arxiv')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
parser.add_argument('--share', type=int, default=0, help='sample_method')
parser.add_argument('--sample_depth', type=int, default=3)
parser.add_argument('--graph_size', type=int, default=100000)

args = parser.parse_args()
print(args)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def temporal_sampler_share(nodes):
    Feats = []
    Adjs = []
    all_sampled_nodes = set(nodes.tolist())
    for x, adj in zip(feats, adjs):
        graph = Data(x=x, edge_index=adj)
        loader = NeighborLoader(graph, input_nodes=nodes, num_neighbors=[-1]*args.sample_depth, batch_size=nodes.shape[0])
        sampled_graph = next(iter(loader))
        sampled_nodes = sampled_graph.n_id
        all_sampled_nodes.update(sampled_nodes.tolist())

    all_sampled_nodes = np.array(list(all_sampled_nodes))
    value_to_index = {value: idx for idx, value in enumerate(all_sampled_nodes)}
    mask = [value_to_index[x] for x in np.array(nodes)]

    all_sampled_nodes = torch.tensor(all_sampled_nodes)
    mask = torch.tensor(mask)

    for x, adj in zip(feats, adjs):
        graph = Data(x=x, edge_index=adj)
        sub_graph = graph.subgraph(all_sampled_nodes)
        feat = sub_graph.x
        edge_index = sub_graph.edge_index
        adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(feat.shape[0], feat.shape[0]))
        adj, feat= utils.to_tensor(adj, feat, device='cpu')
        adj = utils.normalize_adj_tensor(adj, sparse=True)
        adj=SparseTensor(row=adj._indices()[0], col=adj._indices()[1],value=adj._values(), sparse_sizes=adj.size()).t()

        Adjs.append(adj)
        Feats.append(feat)

    return Feats, Adjs, mask

def temporal_sampler(nodes):
    Feats = []
    Adjs = []
    for x, adj in zip(feats, adjs):
        graph = Data(x=x, edge_index=adj)
        loader = NeighborLoader(graph, input_nodes=nodes, num_neighbors=[-1]*args.sample_depth, batch_size=nodes.shape[0])
        sampled_data = next(iter(loader))

        feat = sampled_data.x
        edge_index = sampled_data.edge_index
        adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(feat.shape[0], feat.shape[0]))
        adj, feat= utils.to_tensor(adj, feat, device='cpu')
        if utils.is_sparse_tensor(adj):
            adj = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj = utils.normalize_adj_tensor(adj)
        adj=SparseTensor(row=adj._indices()[0], col=adj._indices()[1],value=adj._values(), sparse_sizes=adj.size()).t()

        Adjs.append(adj)
        Feats.append(feat)
        mask = sampled_data.input_id
    return Feats, Adjs, mask

def split(loader):
    batchs = []
    for nodes in loader:
        if args.share==0:
            Feats, Adjs, mask = temporal_sampler(nodes)
        else:
            Feats, Adjs, mask = temporal_sampler_share(nodes)
        batch = {
        "nodes": nodes,
        "Feats": Feats,              
        "Adjs": Adjs,   
        "mask": mask                  
        }
        batchs.append(batch)
    return batchs

root=os.path.abspath(os.path.dirname(__file__))
data = torch.load(f'./data/processed/{args.dataset}.pt')
device = f'cuda:{args.cuda}'

feats = data.feats
adjs = data.adjs
print(feats.shape)

labels = data.labels.to(device)
idx_train, idx_val, idx_test=data.train_nodes, data.val_nodes, data.test_nodes


train_loader = DataLoader(idx_train.tolist(), pin_memory=False, batch_size=args.graph_size)
val_loader = DataLoader(idx_val.tolist(), pin_memory=False, batch_size=args.graph_size)
test_loader = DataLoader(idx_test.tolist(), pin_memory=False, batch_size=args.graph_size)

num_steps = data.feats.shape[0]
n_nodes = data.feats.shape[1]
d = data.feats.shape[2]
nclass= int(labels.max()+1)

print("splitting!")
train_batch=split(train_loader)
val_batch=split(val_loader)
test_batch=split(test_loader)
if args.share==0:
    torch.save(train_batch, f'{root}/data/splited/'+args.dataset+'_'+'train_batch'+'.pt')
    torch.save(val_batch, f'{root}/data/splited/'+args.dataset+'_'+'val_batch'+'.pt')
    torch.save(test_batch, f'{root}/data/splited/'+args.dataset+'_'+'test_batch'+'.pt')
else:
    torch.save(train_batch, f'{root}/data/splited/'+args.dataset+'_'+'train_batch'+'_share'+'.pt')
    torch.save(val_batch, f'{root}/data/splited/'+args.dataset+'_'+'val_batch'+'_share'+'.pt')
    torch.save(test_batch, f'{root}/data/splited/'+args.dataset+'_'+'test_batch'+'_share'+'.pt')


