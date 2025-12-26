from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import GRU
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import ChebConv
from torch_geometric.nn import TopKPooling

import numpy as np
import random
import time
import argparse
import time
import deeprobust.graph.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import torch_sparse
import os 
import gc
import math

from models.basicgnn import GCN, SGC, GraphSAGE, propagater
from models.spiking_neuron import *
 
from utils import *
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.utils import coalesce
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.init as init
import scipy.sparse as sp
from torch_geometric.nn.dense.linear import Linear



class GCRN(torch.nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, num_timesteps, nlayers=2, dropout=0.5, activation='relu'):
        super(GCRN, self).__init__()
        self.t = num_timesteps
        self.gconv = propagater()
        self.rnn = nn.LSTM(in_features, hidden_feature, num_layers=1)
        self.lin = nn.Linear(hidden_feature, out_features)
        self.norm_out = nn.BatchNorm1d(out_features)
        self.dropout = dropout
        self.nlayers = nlayers

    def forward(self, feats, adjs, edge_weights=None, masks=None):
        embeddings = []
        for time_step in range(self.t):
            x, edge_index = feats[time_step], adjs[time_step]
            edge_weight = edge_weights[time_step] if edge_weights is not None else None
            for i in range(self.nlayers):
                x =self.gconv(x, edge_index, edge_weight) if edge_weight is not None else self.gconv.AX(x, edge_index)
            if masks!=None: 
                embeddings.append(x[masks])
            else:
                embeddings.append(x)

        embedding = torch.stack(embeddings)
        embedding, _ = self.rnn(embedding)
        output = embedding[-1]
        output = self.lin(output)
        output = self.norm_out(output)
        return F.log_softmax(output, dim=1)
     
    @torch.no_grad()
    def predict(self, feats, adjs, edge_weights=None, masks=None):
        self.eval()
        output = self.forward(feats, adjs, edge_weights, masks)
        return output

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()
        self.rnn.reset_parameters()


class TGCN(torch.nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, num_timesteps, nlayers=2, dropout=0.5, activation='relu'):
        super(TGCN, self).__init__()
        self.t = num_timesteps
        self.gconv = nn.ModuleList()
        self.gconv.append(GCNConv(in_features, hidden_feature))
        for _ in range(nlayers-1):
            self.gconv.append(GCNConv(hidden_feature, hidden_feature))
        self.lin = nn.Linear(hidden_feature, out_features)
        self.rnn = nn.LSTM(hidden_feature, hidden_feature, num_layers=1)
        self.dropout = dropout

    def forward(self, feats, adjs, edge_weights=None, masks=None):
        embeddings = []
        for time_step in range(self.t):
            x, edge_index = feats[time_step], adjs[time_step]
            edge_weight = edge_weights[time_step] if edge_weights is not None else None
            for i, layer in enumerate(self.gconv):
                x = layer(x, edge_index, edge_weight) if edge_weight is not None else layer(x, edge_index)
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if masks!=None: 
                embeddings.append(x[masks])
            else:
                embeddings.append(x)

        embedding = torch.stack(embeddings)
        embedding, _ = self.rnn(embedding)
        output = embedding[-1]
        output = self.lin(output)
        return F.log_softmax(output, dim=1)
     
    @torch.no_grad()
    def predict(self, feats, adjs, edge_weights=None, masks=None):
        self.eval()
        output = self.forward(feats, adjs, edge_weights, masks)
        return output

    @torch.no_grad()
    def get_emb(self, feats, adjs, edge_weights=None, masks=None):
        self.eval()
        embeddings = []
        for time_step in range(self.t):
            x, edge_index = feats[time_step], adjs[time_step]
            edge_weight = edge_weights[time_step] if edge_weights is not None else None
            for i, layer in enumerate(self.gconv):
                x = layer(x, edge_index, edge_weight) if edge_weight is not None else layer(x, edge_index)
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if masks!=None: 
                embeddings.append(x[masks])
            else:
                embeddings.append(x)
        embedding = torch.stack(embeddings)
        embedding, _ = self.rnn(embedding)
        output = embedding[-1]
        return output

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.gconv:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.rnn.reset_parameters()


class TGCN_L(torch.nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, num_timesteps, nlayers=2, dropout=0.0, activation='relu'):
        super(TGCN_L, self).__init__()
        self.gcn = GCN(nfeat=in_features, nhid=hidden_feature, nclass=hidden_feature, dropout=dropout, nlayers=nlayers, norm='LayerNorm', act='relu')
        self.rnn = nn.GRU(hidden_feature, out_features, num_layers=1, dropout=dropout)
        self.lin_out = nn.Linear(hidden_feature, out_features)
        self.norm_hid = nn.LayerNorm(hidden_feature)
        self.norm_out = nn.BatchNorm1d(out_features)
        self.dropout = dropout

    def forward(self, feats, adjs, edge_weights=None, masks=None):
        embeddings = []
        steps = len(feats)
        for time_step in range(steps):
            x, edge_index = feats[time_step], adjs[time_step]
            edge_weight = edge_weights[time_step] if edge_weights is not None else None
            emb = self.gcn(x=x, edge_index=edge_index, edge_weight=edge_weight)
            emb = self.norm_hid(emb)
            emb = torch.relu(emb)
            emb = F.dropout(emb, p=self.dropout, training=self.training)
            if masks!=None: 
                embeddings.append(emb[masks])
            else:
                embeddings.append(emb)
        embeddings = torch.stack(embeddings)

        embeddings_rnn, h_n = self.rnn(embeddings)
        output = embeddings_rnn[-1]
        output = self.norm_out(output)
        return F.log_softmax(output,dim=1)

    @torch.no_grad()
    def predict(self, feats, adjs, edge_weights=None, masks=None):
        self.eval()
        output = self.forward(feats, adjs, edge_weights, masks)
        return output
    
    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        self.lin_out.reset_parameters()
        self.rnn.reset_parameters()
        self.norm_hid.reset_parameters()
        self.norm_out.reset_parameters()
        self.gcn.initialize()


class DySAT(torch.nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, num_timesteps, nlayers=2, dropout=0.0):
        super(DySAT, self).__init__()
        self.gconv = nn.ModuleList()
        n_head = 32
        self.gconv.append(GATConv(in_features, hidden_feature//n_head, heads=n_head, dropout=dropout))
        for _ in range(nlayers-1):
            self.gconv.append(GATConv(hidden_feature, hidden_feature//n_head, heads=n_head))
        self.attention = nn.MultiheadAttention(embed_dim=hidden_feature, num_heads=1)
        self.lin_out = nn.Linear(hidden_feature, out_features)
        self.norm_hid = nn.LayerNorm(hidden_feature)
        self.norm_out = nn.BatchNorm1d(out_features)
        self.dropout = dropout

    def forward(self, feats, adjs, edge_weights=None, masks=None):
        embeddings = []
        steps = len(feats)
        for time_step in range(steps):
            x, edge_index = feats[time_step], adjs[time_step]
            edge_weight = edge_weights[time_step] if edge_weights is not None else None
            for i, layer in enumerate(self.gconv):
                x = layer(x, edge_index) if edge_weight is not None else layer(x, edge_index)
            x = self.norm_hid(x)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if masks!=None:
                embeddings.append(x[masks])
            else:
                embeddings.append(x)

        embeddings = torch.stack(embeddings)
        embeddings, _ = self.attention(embeddings, embeddings, embeddings)
        embedding = embeddings[-1]
        output = self.lin_out(embedding)
        output = self.norm_out(output)
        return F.log_softmax(output,dim=1)
    
    @torch.no_grad()
    def predict(self, feats, adjs, edge_weights=None, masks=None):
        self.eval()
        output = self.forward(feats, adjs, edge_weights, masks)
        return output
    
    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        self.lin_out.reset_parameters()
        self.norm_hid.reset_parameters()
        self.norm_out.reset_parameters()
        for conv in self.gconv:
            conv.reset_parameters()
        

class GatedCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(GatedCausalConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        padding = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        x_padded = F.pad(x, (padding, 0))
        conv_out = self.conv(x_padded)
        gate_out = torch.sigmoid(self.gate(x_padded))
        return conv_out * gate_out

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GatedCausalConv(in_channels, out_channels, kernel_size, dilation=1))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class STGCN(torch.nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, num_timesteps, kernel_size=3, nlayers=2, nconv=3, dropout=0.5, activation='relu'):
        super(STGCN, self).__init__()
        self.t = num_timesteps
        self.gconv = propagater()
        self.lin = nn.Linear(in_features, out_features)
        self.tconv = TCN(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, num_layers=nconv)
        self.dropout = dropout
        self.nlayers = nlayers

    def forward(self, feats, adjs, edge_weights=None, masks=None):
        embeddings = []
        for time_step in range(self.t):
            x, edge_index = feats[time_step], adjs[time_step]
            edge_weight = edge_weights[time_step] if edge_weights is not None else None
            for i in range(self.nlayers):
                x =self.gconv(x, edge_index, edge_weight) if edge_weight is not None else self.gconv.AX(x, edge_index)
            if masks!=None: 
                embeddings.append(x[masks])
            else:
                embeddings.append(x)

        embedding = torch.stack(embeddings)
        embedding = embedding.permute(1, 2, 0)
        embedding = self.tconv(embedding).squeeze(dim=2)
        output = self.lin(embedding[:,:,-1])
        
        return F.log_softmax(output, dim=1)
     
    @torch.no_grad()
    def predict(self, feats, adjs, edge_weights=None, masks=None):
        self.eval()
        output = self.forward(feats, adjs, edge_weights, masks)
        return output


    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()


def recursive_moving_average(data, alpha=0.8):
    T, N, D = data.shape
    ema = torch.zeros_like(data).to(data.device)
    ema[0] = data[0] 
    for t in range(1, T):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    return ema

class ROLAND(torch.nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, num_timesteps, nlayers=2, dropout=0.5, activation='relu', update='moving_average'):
        super(ROLAND, self).__init__()
        self.t = num_timesteps
        self.gconv = nn.ModuleList()
        self.gconv.append(GCNConv(in_features, hidden_feature))
        for _ in range(nlayers-1):
            self.gconv.append(GCNConv(hidden_feature, hidden_feature))
        self.bns = nn.ModuleList()
        for _ in range(nlayers):
            self.bns.append(nn.LayerNorm(hidden_feature))
        self.bns.append(nn.LayerNorm(out_features))
        self.lin = nn.Linear(hidden_feature, out_features)
        self.rnn = nn.GRU(hidden_feature, hidden_feature, num_layers=1)
        self.dropout = dropout
        self.nlayers = nlayers
        self.update = update

    def forward(self, feats, adjs, edge_weights=None, masks=None):
        xs = feats
        for i in range(self.nlayers):
            temp = []
            for t in range(self.t):
                x = self.gconv[i](xs[t], adjs[t], edge_weights[t]) if edge_weights is not None else self.gconv[i](xs[t], adjs[t])
                x = self.bns[i](x)
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                temp.append(x)
            xs = torch.stack(temp)
            if self.update=='moving_average':
                xs = recursive_moving_average(xs, alpha=0.8)
            elif self.update=='GRU':
                xs, _ =self.rnn(xs)

        if masks!=None: 
            output = xs[-1][masks]
        else:
            output = xs[-1]
        output = self.lin(output)
        output = self.bns[-1](output)
        return F.log_softmax(output, dim=1)
     
    @torch.no_grad()
    def predict(self, feats, adjs, edge_weights=None, masks=None):
        self.eval()
        output = self.forward(feats, adjs, edge_weights, masks)
        return output

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.gconv:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.rnn.reset_parameters()


