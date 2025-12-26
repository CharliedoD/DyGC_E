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

from models.structure_generation import *
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=3)
parser.add_argument('--dataset', type=str, default='dblp3')
parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
#gnn
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--activation', type=str, default='relu')

parser.add_argument('--test_model', type=str, default='TGCN')
parser.add_argument('--reduction_rate', type=float, default=0.1)
parser.add_argument('--lr_model', type=float, default=0.005)
parser.add_argument('--test_loop', type=int, default=1000)
parser.add_argument('--val_stage', type=int, default=50)

args = parser.parse_args()
print(args)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
    
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

    feats = data.feats.to(device)
    labels = torch.LongTensor(data.labels).to(device)
    n_nodes = feats.shape[1]

    if args.dataset=='reddit':
        feats = F.normalize(feats, p=2, dim=2)
        
    adjs = []
    for edge_index, feat in zip(data.adjs, data.feats):
        edge_weight = torch.ones(edge_index.size(1))
        adj = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=(n_nodes, n_nodes))
        adj = utils.normalize_adj_tensor(adj, sparse=True)
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],value=adj._values(), sparse_sizes=adj.size()).t()
        adjs.append(adj.to(device))

    idx_train, idx_val, idx_test=data.train_nodes, data.val_nodes, data.test_nodes
    feat_train=feats[:,idx_train,:]
    labels_train, labels_val, labels_test=labels[idx_train], labels[idx_val], labels[idx_test]

    nclass= int(labels.max()+1)
    num_steps = feats.shape[0]
    d = feats.shape[-1]


    if args.test_model == 'STGCN':
        test_model = STGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, nconv=3, dropout=args.dropout).to(device)
    elif args.test_model == 'TGCN':
        test_model = TGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'TGCN_L':
        test_model = TGCN_L(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'DySAT':
        test_model = DySAT(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.test_model == 'ROLAND':
        test_model = ROLAND(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout, update='GRU').to(device)
    elif args.test_model == 'GCRN':
        test_model = GCRN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    else:
        print('no model!')
    optimizer_test = optim.Adam(test_model.parameters(), lr=args.lr_model)

    print("testing!")
    feat_syn = torch.load(f'{root}/syn/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    adjs_syn = torch.load(f'{root}/syn/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    labels_syn = torch.load(f'{root}/syn/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt').to(device)
    nnodes_syn = len(labels_syn)

    best_val=0
    best_test=0
    edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, nnodes_syn)

    #test_model.initialize()
    for j in tqdm(range(1, args.test_loop+1), desc="Testing"): 
        test_model.train()
        optimizer_test.zero_grad()
        output_syn = test_model.forward(feat_syn, edge_indexs_syn, edge_weights_syn)
        loss = F.nll_loss(output_syn, labels_syn)
        loss.backward()
        optimizer_test.step()
        if j%args.val_stage==0:
            output = test_model.predict(feats, adjs)
            y_val_pred = output[idx_val].argmax(1)  # Predicted labels for validation set
            y_test_pred = output[idx_test].argmax(1)  # Predicted labels for test set

            f1_val_micro = f1_score(labels_val.cpu(), y_val_pred.cpu(), average='micro')
            f1_test_micro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='micro')

            f1_val_macro = f1_score(labels_val.cpu(), y_val_pred.cpu(), average='macro')
            f1_test_macro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='macro')

            if(f1_val_micro>=best_val):
                best_val=f1_val_micro
                best_test_micro=f1_test_micro
                best_test_macro=f1_test_macro
            print("loop", f'{j}:', "Test micro:", f'{f1_test_micro*100:.2f}', "Test macro::", f'{f1_test_macro*100:.2f}')
    print("Best Test Micro:", f'{best_test_micro*100:.2f}', "Best Test Macro::", f'{best_test_macro*100:.2f}')

    # with open(f'result/{args.dataset}.txt', "a") as result_file:
    #     result_file.write( f"Reduction Rate: {args.reduction_rate} "  f"Teacher model: {args.teacher_model} "
    #                        f"Test model: {args.test_model} "  f"Best Test Micro: {best_test_micro*100:.2f} "  f"Best Test Macro: {best_test_macro*100:.2f} \n")