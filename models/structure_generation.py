import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import product
import numpy as np
import deeprobust.graph.utils as utils
from utils import *
from models.spiking_neuron import *

class PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.args = args
        self.nnodes = nnodes
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x, inference=False):
        edge_index = self.edge_index
        source, target = x[edge_index[0]], x[edge_index[1]]
        edge_embed = torch.einsum('ij,ij->i', source, target).unsqueeze(-1)
        edge_embed = self.norm(edge_embed)
        adj = edge_embed.reshape(self.nnodes, self.nnodes)
        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)


class SNN_PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, n_steps=10, device=None, args=None):
        super(SNN_PGE, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))
        self.bns.append(nn.BatchNorm1d(1))

        self.SNN = LIF(nnode=nnodes, n_steps=n_steps)
        self.SNN.reset()
        self.norm = nn.BatchNorm1d(1)

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.args = args
        self.nnodes = nnodes

    def forward(self, x, inference=False):
        edge_index = self.edge_index
        source, target = x[edge_index[0]], x[edge_index[1]]
        # edge_embed = torch.cat([source, target], axis=1).to(self.device)
        # for ix, layer in enumerate(self.layers):
        #     edge_embed = layer(edge_embed)
        #     edge_embed = self.bns[ix](edge_embed)
        #     edge_embed = F.relu(edge_embed)

        edge_embed = torch.einsum('ij,ij->i', source, target).unsqueeze(-1)
        #edge_embed = F.cosine_similarity(source, target, dim=1).unsqueeze(-1)
        edge_embed = self.norm(edge_embed)
        adj = edge_embed.reshape(self.nnodes, self.nnodes)
        adj = adj + adj.T
        adj = self.SNN(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

class SNN_generator(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, n_steps=10, args=None):
        super(SNN_generator, self).__init__()
        
        self.pge = SNN_PGE(nfeat=nfeat, nnodes=nnodes, nhid=nhid, nlayers=nlayers, device=device, args=args, n_steps=n_steps)
        self.nnodes = nnodes
        self.args = args
        self.device=device
        self.layers = nn.ModuleList([])

    def forward(self, feats, inference=False):
        adjs_syn = []
        self.pge.SNN.reset()
        for idx, feat in enumerate(feats):
            if inference:
                adj_syn = self.pge(feat).detach().to(self.device)
                adj_syn.requires_grad=False
            else:
                adj_syn = self.pge(feat).to(self.device)
            adjs_syn.append(adj_syn)
        adjs_syn = torch.stack(adjs_syn, dim=0)
        return adjs_syn

class adj_generator(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, n_steps=10, args=None):
        super(adj_generator, self).__init__()
        
        self.pge = PGE(nfeat=nfeat, nnodes=nnodes, nhid=nhid, nlayers=nlayers, device=device, args=args)
        self.nnodes = nnodes
        self.args = args
        self.device=device
        self.layers = nn.ModuleList([])

    def forward(self, feats, inference=False):
        adjs_syn = []
        for feat in feats:
            if inference:
                adj_syn = self.pge(feat).detach().to(self.device)
                adj_syn.requires_grad=False
            else:
                adj_syn = self.pge(feat).to(self.device)
            adj_syn[adj_syn<0.01]=0
            adjs_syn.append(adj_syn)
        adjs_syn = torch.stack(adjs_syn, dim=0)

        return adjs_syn


class adj_generator_ind(nn.Module):
    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(adj_generator_ind, self).__init__()
        
        self.pge = PGE(nfeat=nfeat, nnodes=nnodes, nhid=nhid, nlayers=nlayers, device=device, args=args)
        self.nnodes = nnodes
        self.args = args
        self.device=device

    def forward(self, feat, inference=False):
        if inference:
            adj_syn = self.pge(feat).detach().to(self.device)
            adj_syn.requires_grad=False
        else:
            adj_syn = self.pge(feat).to(self.device)
        adj_syn[adj_syn<0.01]=0
        edge_index_syn = torch.nonzero(adj_syn).T
        edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]
        edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, self.nnodes)

        return edge_index_syn, edge_weight_syn