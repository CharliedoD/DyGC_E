from models.DGNN import *
import torch
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
from torch_geometric.nn import MessagePassing

from models.basicgnn import propagater
from models.structure_generation import *
from sklearn.metrics import f1_score


from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, add_self_loops

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--seed', type=int, default=2025, help='Random seed.')

parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--K', type=int, default=2)

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--teacher_model', type=str, default='SGC_MLP')
parser.add_argument('--val_model', type=str, default='DGCN')

parser.add_argument('--reduction_rate', type=float, default=0.1)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.005)

parser.add_argument('--loss_factor', type=float, default=10, help='feat loss term.')
parser.add_argument('--temporal_alpha', type=float, default=0.1, help='sim loss term.')

parser.add_argument('--condensing_loop', type=int, default=200)
parser.add_argument('--teacher_training_loop', type=int, default=1000)
parser.add_argument('--condensing_val_stage', type=int, default=10)
parser.add_argument('--student_model_loop', type=int, default=1000)
parser.add_argument('--student_val_stage', type=int, default=50)
args = parser.parse_args()
print(args)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#torch.use_deterministic_algorithms(True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def generate_labels_syn():
    from collections import Counter
    counter = Counter(labels_train.cpu().numpy())
    num_class_dict = {}

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    labels_syn = []
    syn_class_indices = {}

    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * args.reduction_rate)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

    return labels_syn, num_class_dict


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


def train_teacher():
    patience = 150  # 容忍验证集性能未提升的次数
    stopping_counter = 0 
    best_val=0
    best_test_micro=0
    best_test_macro=0
    optimizer=torch.optim.Adam(teacher_model.parameters(), lr=args.lr_model)

    for epoch in tqdm(range(1, args.teacher_training_loop+1), desc="Training Teacher"): 
        teacher_model.train()
        optimizer.zero_grad()
        output = teacher_model.forward(feats, adjs)
        loss = F.nll_loss(output[idx_train], labels_train)
        loss.backward()
        optimizer.step()

        output_predict = teacher_model.predict(feats, adjs)
        
        y_val_pred = output_predict[idx_val].argmax(1)  # Predicted labels for validation set
        y_test_pred = output_predict[idx_test].argmax(1)  # Predicted labels for test set

        f1_val_micro = f1_score(labels_val.cpu(), y_val_pred.cpu(), average='micro')
        f1_test_micro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='micro')

        f1_val_macro = f1_score(labels_val.cpu(), y_val_pred.cpu(), average='macro')
        f1_test_macro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='macro')

        if epoch % 50 == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss.item():.4f}, '
                f'Best Test Micro-F1: {100 * best_test_micro:.2f}%  '
                f'Best Test Macro-F1: {100 * best_test_macro:.2f}%  '
                )

        if f1_val_micro > best_val:
            best_val = f1_val_micro
            best_test_micro = f1_test_micro
            best_test_macro = f1_test_macro
            stopping_counter = 0
            torch.save(teacher_model.state_dict(), f'{root}/teacher/{args.teacher_model}_{args.dataset}_{args.dropout}_{args.seed}.pt')
        else:
            stopping_counter += 1

        if stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs!")
            break 

def train_syn():
    start = time.perf_counter()

    optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
    optimizer_pge = optim.Adam(Structure_generator.parameters(), lr=args.lr_adj)

    concat_feat_all = []
    agg_memory = []
    alpha = args.temporal_alpha
    beta = 0.0
    for time_step in range(num_steps):
        concat_feat=feat_train[time_step]
        temp=feats[time_step].to(device)
        adj = adjs[time_step]

        for k in range(args.K):
            aggr = propagater(edge_index=adj, x=temp, edge_weight=None) 
            if time_step==0:
                agg_memory.append(aggr)
            else:
                aggr = alpha * agg_memory[k] + (1-alpha) * aggr
                agg_memory[k] = aggr                
            concat_feat=torch.cat((concat_feat, aggr[idx_train]),dim=1)
            temp=aggr        
        concat_feat_all.append(concat_feat)
    concat_feat_all = torch.stack(concat_feat_all)
    FEAT_org = concat_feat_all.permute(1, 0, 2)


    feats_org=[]
    coeff=[]
    coeff_sum=0
    for c in range(nclass):
        if c in num_class_dict:
            index = torch.where(labels_train==c)[0].to('cpu')
            coe = num_class_dict[c] / max(num_class_dict.values())
            coeff_sum+=coe
            coeff.append(coe)           
            feat_org_c = FEAT_org[index]
            feats_org.append(feat_org_c)
    coeff_sum=torch.tensor(coeff_sum).to(device)


    best_val=0
    best_test_micro=0
    best_test_macro=0
    loss_fn_MMD=MMDLoss()

    for i in tqdm(range(1, args.condensing_loop+1), desc="Condensing"):

        adjs_syn = Structure_generator(feat_syn)
        edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, nnodes_syn)

        concat_feat_syn_all=[]
        aggr_syn_memory = []
        for time_step in range(num_steps):
            concat_feat_syn=feat_syn[time_step]
            temp=feat_syn[time_step]
            edge_index_syn = edge_indexs_syn[time_step]
            edge_weight_syn = edge_weights_syn[time_step]
            for k in range(args.K):
                aggr_syn = propagater(edge_index=edge_index_syn, x=temp, edge_weight=edge_weight_syn)
                if time_step==0:
                    aggr_syn_memory.append(aggr_syn)
                else:
                    aggr_syn = alpha * aggr_syn_memory[k] + (1-alpha) * aggr_syn
                    aggr_syn_memory[k] = aggr_syn    
                concat_feat_syn=torch.cat((concat_feat_syn, aggr_syn),dim=1)
                temp=aggr_syn
            concat_feat_syn_all.append(concat_feat_syn)
        concat_feat_syn_all = torch.stack(concat_feat_syn_all)
        FEAT_syn = concat_feat_syn_all.permute(1, 0, 2)

        feats_syn = []
        for c in range(nclass):
            if c in num_class_dict:
                index = torch.where(labels_syn==c)[0]
                feats_syn.append(FEAT_syn[index])
    
        mmd_loss=torch.tensor(0.0).to(device)
        for c in range(nclass):
            if c in num_class_dict:
                index=torch.where(labels_syn==c)[0]
                mmd_loss+=coeff[c]*loss_fn_MMD(feats_org[c], feats_syn[c])
        mmd_loss=mmd_loss/coeff_sum

        teacher_model.train()
        output_syn = teacher_model.forward(feat_syn, edge_indexs_syn, edge_weights_syn)
        hard_loss = F.nll_loss(output_syn, labels_syn)
        
        loss = hard_loss + args.loss_factor*mmd_loss
        
        optimizer_pge.zero_grad()
        optimizer_feat.zero_grad()
        loss.backward() 
        optimizer_pge.step()
        optimizer_feat.step()

        
        if i>=100 and i%args.condensing_val_stage==0:
            adjs_syn = Structure_generator(feat_syn, inference=True)
            edge_indexs_syn, edge_weights_syn = gcn_norm_temporal(adjs_syn, nnodes_syn)
                
            val_model.initialize()
            teacher_output_syn = teacher_model.predict(feat_syn, edge_indexs_syn, edge_weights_syn)
            acc = utils.accuracy(teacher_output_syn, labels_syn)
            print('Epoch {}'.format(i),"Teacher on syn accuracy= {:.2f}".format(100*acc.item()))

            test_micros, test_macros = [], []
            val_micros, val_macros = [], []
            for j in range(args.student_model_loop):
                val_model.train()
                optimizer.zero_grad()
                output_syn = val_model.forward(feat_syn, edge_indexs_syn, edge_weights_syn )
                loss = F.nll_loss(output_syn, labels_syn)
                loss.backward()
                optimizer.step()
                if j % args.student_val_stage == 0:
                    output = val_model.predict(feats.to(device), adjs)
                    y_val_pred = output[idx_val].argmax(1)  
                    y_test_pred = output[idx_test].argmax(1)  

                    f1_val_micro = f1_score(labels_val.cpu(), y_val_pred.cpu(), average='micro')
                    f1_test_micro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='micro')

                    f1_val_macro = f1_score(labels_val.cpu(), y_val_pred.cpu(), average='macro')
                    f1_test_macro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='macro')

                    test_micros.append(f1_test_micro)
                    test_macros.append(f1_test_macro)

                    if f1_val_micro >= best_val:
                        best_val = f1_val_micro
                        best_test_micro = f1_test_micro
                        best_test_macro = f1_test_macro
                        torch.save(feat_syn, f'{root}/syn/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                        torch.save(adjs_syn, f'{root}/syn/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                        torch.save(labels_syn, f'{root}/syn/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')


            print("test Micro:", ', '.join([f"{tensor*100:.2f}" for tensor in test_micros]))
            print("test Macro:", ', '.join([f"{tensor*100:.2f}" for tensor in test_macros]))

            print(f'Epoch {i} | Best test Micro F1: {100 * best_test_micro:.2f}%  Best test Macro F1: {100 * best_test_macro:.2f}%')

    end = time.perf_counter()
    print('Condensation Duration:',round(end-start), 's')


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

    nclass= int(labels.max()+1)
    num_steps = feats.shape[0]
    d = feats.shape[-1]

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

    print(feat_train.shape)
    #########################################################################################################################################################################
    if args.teacher_model == 'GCRN':
        teacher_model = GCRN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.teacher_model == 'ROLAND':
        teacher_model = ROLAND(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.teacher_model == 'DySAT':
        teacher_model = DySAT(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.teacher_model == 'TGCN':
        teacher_model = TGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.teacher_model == 'STGCN':
        teacher_model = STGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    else:
        print('no model!')

    if not os.path.exists(f'{root}/teacher/{args.teacher_model}_{args.dataset}_{args.dropout}_{args.seed}.pt'):
        train_teacher()
    teacher_model.load_state_dict(torch.load(f'{root}/teacher/{args.teacher_model}_{args.dataset}_{args.dropout}_{args.seed}.pt'))
    teacher_model.eval()
 
    output = teacher_model.predict(feats.to(device), adjs)
    y_test_pred = output[idx_test].max(dim=1)[1] 
    f1_test_micro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='micro')
    f1_test_macro = f1_score(labels_test.cpu(), y_test_pred.cpu(), average='macro')

    print(f'Test Micro: {100 * f1_test_micro:.2f}%  '
         f'Test Macro: {100 * f1_test_macro:.2f}%')
    ###############################################################################################################################################################################
    if args.val_model == 'GCRN':
        val_model = GCRN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.val_model == 'ROLAND':
        val_model = ROLAND(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.val_model == 'DySAT':
        val_model = DySAT(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.val_model == 'TGCN':
        val_model = TGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    elif args.val_model == 'STGCN':
        val_model = STGCN(in_features=d, hidden_feature=args.hidden, out_features=nclass, num_timesteps=num_steps, nlayers=args.nlayers, dropout=args.dropout).to(device)
    else:
        print('no model!')
    optimizer = optim.Adam(val_model.parameters(), lr=args.lr_model)
    ################################################################################################################################################################################
    propagater = propagater().to(device)
    ################################################################################################################################################################################
    labels_syn, num_class_dict = generate_labels_syn()
    labels_syn = torch.LongTensor(labels_syn).to(device)
    nnodes_syn = len(labels_syn)
    n = nnodes_syn

    feat_syn = nn.Parameter(torch.FloatTensor(num_steps, n, d).to(device))
    feat_syn.data.copy_(torch.randn(feat_syn.size()))
    print('size of feat_syn: '+ str(feat_syn.shape))

    Structure_generator = SNN_generator(nfeat=d, nnodes=n, n_steps=num_steps, args=args).to(device)
    ################################################################################################################################################################################

    print("Condensing!")
    train_syn()
