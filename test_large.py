from models.DGNN import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import argparse
import time
import scipy.sparse as sp
import deeprobust.graph.utils as utils
import os 
import gc
import math
import random
import numpy as np
from tqdm import tqdm

from utils import *
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.utils import coalesce

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='dblp5')
parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
parser.add_argument('--testseed', type=int, default=2025, help='Random seed.')
#gnn
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--activation', type=str, default='relu')

parser.add_argument('--test_model', type=str, default='TGCN_L')
parser.add_argument('--reduction_rate', type=float, default=0.1)
parser.add_argument('--lr_model', type=float, default=0.001)

parser.add_argument('--test_loop', type=int, default=2000)
parser.add_argument('--val_stage', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100000)
args = parser.parse_args()
print(args)

random.seed(args.testseed)
np.random.seed(args.testseed)
torch.manual_seed(args.testseed)
torch.cuda.manual_seed(args.testseed)

def to_device(batch, device='cuda:0'):
    nodes = batch["nodes"].to(device)
    Feats = [feat.to(device) for feat in batch["Feats"]]
    Adjs = [adj.to(device) for adj in batch["Adjs"]]
    mask = batch["mask"].to(device)
    batch_in_device = {
    "nodes": nodes,
    "Feats": Feats,              
    "Adjs": Adjs,   
    "mask": mask                  
    }
    return batch_in_device

def test(input_batch, val_model):
    Logits = []
    Labels = []
    for batch in input_batch:
        batch = to_device(batch, device)
        output = val_model.predict(feats=batch["Feats"], adjs=batch["Adjs"], masks=batch["mask"])
        Logits.append(output)
        Labels.append(labels[batch["nodes"]])
    Logits = torch.cat(Logits, dim=0).argmax(1)
    Labels = torch.cat(Labels, dim=0)
    micro_f1 = f1_score(Logits.cpu(), Labels.cpu(), average='micro')
    macro_f1 = f1_score(Logits.cpu(), Labels.cpu(), average='macro')
    return micro_f1, macro_f1


def gcn_norm_temporal(adjs, nnodes):
    edge_indexs=[]
    edge_weights=[]
    for adj in adjs:
        edge_index = torch.nonzero(adj).T
        edge_weight = adj[edge_index[0], edge_index[1]]
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, nnodes)
        edge_indexs.append(edge_index)
        edge_weights.append(edge_weight)
    return edge_indexs, edge_weights



class GraphData:
    def __init__(self, feats, labels, adjs, train_nodes, val_nodes, test_nodes):
        self.feats = feats
        self.labels = labels
        self.adjs = adjs
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes

    
if __name__ == '__main__':
    root=os.path.abspath(os.path.dirname(__file__))
    data = torch.load(f'./data/processed/{args.dataset}.pt')
    device = f'cuda:{args.cuda}'

    feats = data.feats
    adjs = data.adjs

    labels = torch.LongTensor(data.labels).to(device)
    idx_train, idx_val, idx_test=data.train_nodes, data.val_nodes, data.test_nodes
    labels_train, labels_val, labels_test=labels[idx_train], labels[idx_val], labels[idx_test]

    num_steps = data.feats.shape[0]
    n_nodes = data.feats.shape[1]
    d = data.feats.shape[2]
    nclass= int(labels.max()+1)

    train_batch=torch.load(f'{root}/data/splited/'+args.dataset+'_'+'train_batch'+'.pt')
    val_batch=torch.load(f'{root}/data/splited/'+args.dataset+'_'+'val_batch'+'.pt')
    test_batch=torch.load(f'{root}/data/splited/'+args.dataset+'_'+'test_batch'+'.pt')

    if args.test_model == 'TGCN_L':
        test_model = TGCN_L(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'TGCN':
        test_model = TGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'STGCN':
        test_model = STGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'DySAT':
        test_model = DySAT(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'GCRN':
        test_model = GCRN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'ROLAND':
        test_model = ROLAND(in_channels=d, out_channels=nclass, hidden_channels=args.hidden, num_layers=args.nlayers, ssm_format='siso', token_mixer='conv1d', d_state=16).to(device)
    else:
        print('no chosen model')
    optimizer_test = optim.AdamW(test_model.parameters(), lr=args.lr_model)


    test_model.initialize()    
    print("testing!")
    feat_syn = torch.load(f'{root}/syn/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    adjs_syn = torch.load(f'{root}/syn/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    labels_syn = torch.load(f'{root}/syn/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    nnodes_syn = len(labels_syn)


    best_val=0
    best_test_micro=0
    best_test_macro=0
    edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, nnodes_syn)

    for epoch in range(1, args.test_loop+1):    
        test_model.train()
        output_syn = test_model.forward(feat_syn, edge_indexs_syn, edge_weights_syn)
        loss = F.nll_loss(output_syn, labels_syn)
        optimizer_test.zero_grad()
        loss.backward()
        optimizer_test.step()

        if epoch%args.val_stage==0:
            val_micro, val_macro = test(val_batch, test_model)
            test_micro, test_macro= test(test_batch, test_model)
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Test Micro: {100 * test_micro:.2f}% '
                    f'Test Macro: {100 * test_macro:.2f}% '
                    )
            if(val_micro>best_val):
                best_val=val_micro
                best_test_micro=test_micro
                best_test_macro=test_macro

    print(  f'Best Test Micro: {100 * best_test_micro:.2f}% '
            f'Best Test Macro: {100 * best_test_macro:.2f}% '
            )
