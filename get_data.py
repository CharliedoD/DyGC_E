import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from torch_sparse import SparseTensor
import deeprobust.graph.utils as utils
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='dblp')
args = parser.parse_args()


name = args.data
file = np.load('./data/raw/' + str(name) + '.npz')

feats = file['attmats']  # (N, T, D) node features
labels = file['labels']  # (N, C) node labels
adjs = file['adjs']  # (T, N, N) graph snapshots

# Convert to PyTorch tensors
feats = torch.tensor(feats, dtype=torch.float32).transpose(0, 1)
if args.data == 'reddit':
    feats = F.normalize(feats, p=2, dim=2)
labels = torch.tensor(labels, dtype=torch.float32)
labels = torch.argmax(labels, dim=1)

#adjs = torch.tensor(adjs, dtype=torch.float32)
def adjacency_to_edge_index(adj):
    # 将邻接矩阵转换为边索引
    #adj = np.array(adj)  # 确保输入是 NumPy 数组
    edge_index = np.nonzero(adj)  # 获取非零元素的索引
    edge_index = np.array(edge_index)  # 转换为 NumPy 数组
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # 转换为 PyTorch 张量
    return edge_index

edge_indexs = []
for adj in adjs:
    adj = adjacency_to_edge_index(adj)
    print(adj.shape)
    edge_indexs.append(adj)

#sparse_adjs = torch.stack(sparse_adjs,)
adjs = [sp.csr_matrix(adj) for adj in adjs]

train_nodes, test_nodes = train_test_split(
    torch.arange(labels.size(0)),
    train_size=0.5,
    test_size=0.5,
    random_state=42,
    stratify=labels) 


val_nodes, test_nodes = train_test_split(
    test_nodes,
    train_size=0.2 / 0.5,
    random_state=42,
    stratify=labels[test_nodes])

class GraphData:
    def __init__(self, feats, labels, adjs, train_nodes, val_nodes, test_nodes):
        self.feats = feats
        self.labels = labels
        self.adjs = edge_indexs
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes

    def save(self, filename):
        torch.save(self, filename)
        print(f"GraphData saved to {filename}")

graph_data = GraphData(
    feats,
    labels,
    adjs,
    train_nodes,
    val_nodes,
    test_nodes
)

# 保存到 .pt 文件
graph_data.save('./data/processed/' + str(name) + '.pt')

print(train_nodes.shape)
print(val_nodes.shape)
print(test_nodes.shape)
