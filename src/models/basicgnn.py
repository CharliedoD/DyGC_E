import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm
from torch_sparse import SparseTensor, matmul

from torch_geometric.loader import NeighborSampler

from .convs import (
    EdgeConv,
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    SGConv,
)

from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor, PairTensor

class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        nfeat (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        nhid (int): Size of each hidden sample.
        nlayers (int): Number of message passing layers.
        nclass (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`nclass`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nlayers: int,
        nclass: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        sgc: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.temp_layers = nlayers
        if sgc==False:
            self.nlayers = nlayers
        else:
            self.nlayers = 1    
            nlayers = 1

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs
        self.sgc=sgc

        if nclass is not None:
            self.nclass = nclass
        else:
            self.nclass = nhid

        self.convs = ModuleList()
        if nlayers > 1:
            self.convs.append(
                self.init_conv(nfeat, nhid, **kwargs))
            if isinstance(nfeat, (tuple, list)):
                nfeat = (nhid, nhid)
            else:
                nfeat = nhid
        for _ in range(nlayers - 2):
            self.convs.append(
                self.init_conv(nfeat, nhid, **kwargs))
            if isinstance(nfeat, (tuple, list)):
                nfeat = (nhid, nhid)
            else:
                nfeat = nhid
        if nclass is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(nfeat, nclass, **kwargs))
        else:
            self.convs.append(
                self.init_conv(nfeat, nhid, **kwargs))

        self.norms = None
        if norm is not None and sgc==False:
            self.norms = ModuleList()
            for _ in range(nlayers-1):
                norm_layer = normalization_resolver(norm, nhid, **(norm_kwargs or {}),)
                self.norms.append(copy.deepcopy(norm_layer))
            norm_layer = normalization_resolver(norm, nclass, **(norm_kwargs or {}),)
            self.norms.append(copy.deepcopy(norm_layer))


    def init_conv(self, nfeat: Union[int, Tuple[int, int]],
                  nclass: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def initialize(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
        """
        for i in range(self.nlayers):
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.nlayers - 1 and self.jk_mode is None:
                break
            if self.norms is not None:
                x = self.norms[i](x)
            # if self.act is not None:
            #     x = self.act(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    
    def forward_sampler(self, x, adjs):
        if self.sgc==False:
            for i, (adj, _, size) in enumerate(adjs):
                x = self.convs[i](x, adj)
                if i != self.nlayers - 1:
                    x = self.norms[i](x)
                    x = self.act(x)
                    x = F.dropout(x, self.dropout, training=self.training)
        else:
            x = self.convs[0].forward_sampler(x, adjs)
        
        return x
        
    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:

        self.eval()
        return self.forward(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, nlayers={self.nlayers})')


class GCN(BasicGNN):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(nfeat, nclass, **kwargs)


class GraphSAGE(BasicGNN):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: Union[int, Tuple[int, int]],
                  nclass: int, **kwargs) -> MessagePassing:
        return SAGEConv(nfeat, nclass, project=False, **kwargs)

class SGC(BasicGNN):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, nfeat: int, nclass: int,
                  **kwargs) -> MessagePassing:
        return SGConv(nfeat, nclass, self.temp_layers, **kwargs)
    



class propagater(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, K={self.K})')

