
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import imp
from copy import deepcopy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid,Reddit2,Flickr
from help_funcs import reconstruct_prune_unrelated_edge
# from models.LogReg import LogRegs
# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv', 'Citeseer', 'Physics'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)

parser.add_argument('--target_class', type=int, default=0)

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")

parser.add_argument('--vs_number', type=int, default=40,
                    help="number of poisoning nodes relative to the full graph")

# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none', 'reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.5,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")

parser.add_argument('--homo_loss_weight', type=float, default=50,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5, 
                    help="Threshold of increase similarity")

parser.add_argument('--isolation_ratio', type=float, default=0.01,
                    help="ABL isolation ratio")
parser.add_argument('--isolate_epoch', type=int, default=20,
                    help="ABL isolate epoch")
parser.add_argument('--loss_increse_threshold', type=float, default=1.4,
                    help="ABL loss_increse_threshold") 
parser.add_argument('--threshold', type=float, default=0.5,
                    help="ABL threshold") 

parser.add_argument('--split_lr', type=float, default=0.03,
                    help='split learning rate.')
parser.add_argument('--split_epoch', type=int, default=20,
                    help="split target nodes")
 

# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='cluster_degree',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN','DGI', 'GNNGuard','RobustGCN', 'ABL'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='overall',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")

parser.add_argument('--rec_epochs', type=int,  default=100,
                    help='Number of epochs to train benign and backdoor model.')

parser.add_argument('--trigger_generator_address', type=str, default='./weights/UGBA/Cora/UGBA_Cora_weights.pth')
parser.add_argument('--pre_train_param', type=str, default='./weights/UGBA/Cora/UGBA_Cora.pt')
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 
elif(args.dataset == 'Physics'):
    from torch_geometric.datasets import Coauthor
    dataset = Coauthor(root='./data/Physics', name='Physics')


data = dataset[0].to(device)


if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
elif(args.dataset=='Physics'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)

# we build our own train test split 
#%% 
from utils import get_split

data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args, data, device)



from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
print(data.edge_index.shape)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


# In[9]:

from sklearn_extra import cluster
from models.backdoor_UGBA import Backdoor
from models.construct import model_construct
import heuristic_selection as hs


# from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
print(unlabeled_idx.shape)
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("#Attach Nodes:{}".format(size))
assert size>0, 'The number of selected trigger nodes must be larger than 0!'
# here is randomly select poison nodes from unlabeled nodes
if(args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    if(args.dataset == 'Pubmed'):
        idx_attach = hs.cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    else:
        idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
mask = data.y[idx_attach] != args.target_class
mask = mask.to(device)
idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
print("idx_attach: {}".format(idx_attach))
print("num_attach: ", len(idx_attach))
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
size = args.vs_number
from models.construct import model_construct
if(args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
mask = data.y[idx_attach] != args.target_class
mask = mask.to(device)
idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
print("idx_attach: {}".format(idx_attach))
print("num_attach: ", len(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)

known_nodes = torch.cat([idx_train,idx_attach]).to(device)

print(data)

###LoSplit Defense###
LoSplit= model_construct(args,'LoSplit',data,device).to(device) 
train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
poison_found, clean_found, deltas, target_labels = LoSplit.fit(data.x, train_edge_index, train_edge_weights, data.y, known_nodes, idx_val,train_iters=args.epochs,verbose=False, split_epoch=args.split_epoch)


best_delta = max(deltas[1:])  
best_t = 1
idx_poison_found = poison_found[best_t]
idx_clean_found = clean_found[best_t]
target_label = target_labels[best_t]



test_model = model_construct(args,args.test_model,data,device).to(device) 


test_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)

test_model.eval()

output = test_model(data.x,data.edge_index, None)

clean_acc = test_model.test(data.x,data.edge_index, None, data.y,idx_clean_test)

print("****After LoSplit Defense (clean dataset)****")

print("Clean Accuracy: {:.6f}".format(clean_acc)) 

poison_common = set(idx_poison_found.tolist()) & set(idx_attach.tolist())
clean_common = set(idx_clean_found.tolist()) & set(idx_train.tolist())

TP = len(poison_common)   # 正确检测出的中毒节点
FP = len(idx_poison_found) - TP  # 被误判为中毒节点的干净节点
FN = len(idx_attach) - TP  # 没有被检测出的中毒节点
TN = len(clean_common) # 正确识别的干净节点

fpr = FP / (FP + TN + 1e-8)

print(f"FPR: {fpr:.4f}")