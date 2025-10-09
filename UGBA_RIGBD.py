
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.stats import norm
from torch.distributions.bernoulli import Bernoulli
from collections import Counter
from sklearn.metrics import pairwise_distances
import pandas as pd

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams   # ✅ 这样才能用 rcParams
rcParams['font.sans-serif'] = ['Songti SC']   # 支持中文
rcParams['font.family'] = 'sans-serif'
font_cn = FontProperties(family="Songti SC", size=18)      # 中文宋体
font_en = FontProperties(family="Times New Roman", size=20) # 英文 + 数字
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 25            

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
parser.add_argument('--train_lr', type=float, default=0.012,
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
parser.add_argument('--lr', type=float, default=0.012877,
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
parser.add_argument('--prune_thr', type=float, default=0.2,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")

parser.add_argument('--homo_loss_weight', type=float, default=50,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5, 
                    help="Threshold of increase similarity")
 

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
parser.add_argument('--pre_train_param', type=str, default='./weights/UGBA/Cora/UGBA_Cora_RIGBD.pt')
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
elif(args.dataset == 'Physics'):
    from torch_geometric.datasets import Coauthor
    dataset = Coauthor(root='./data/Physics', name='Physics')
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 


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
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
# In[10]:

# train trigger generator 
model = Backdoor(args,device)

# model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
# torch.save(model.trojan.state_dict(), args.trigger_generator_address)

model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx, args.trigger_generator_address, True)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()


known_nodes = torch.cat([idx_train,idx_attach]).to(device)
# edge weight for clean edge_index, may use later #
edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)

if(args.defense_mode == 'prune'):
    poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'reconstruct'):
    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,data.x,data.edge_index,device, idx_attach, large_graph=True)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)

elif(args.defense_mode == 'isolate'):
    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))

test_model = model_construct(args,args.test_model,data,device).to(device) 

if test_model == 'ABL':
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, num_attach=len(idx_attach))
else:
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)

output = test_model(poison_x, poison_edge_index, poison_edge_weights)
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
# output = test_model(induct_x,induct_edge_index,induct_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)

output = test_model(induct_x,induct_edge_index,induct_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()

asr = train_attach_rate
flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()

print("****After UGBA Attack****")
print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
print("Clean Accuracy: {:.6f}".format(clean_acc)) 
induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
output = output.cpu()


# RIGBD Defense

K_list = [25, 50, 100]
drop_ratio_list = [0.3, 0.5, 0.7, 0.9]
results = {}

for drop_ratio in drop_ratio_list:
    for K in K_list:
        test_model = model_construct(args,args.test_model,data,device, add_self_loops=False).to(device) 
        test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        test_model.eval()
        clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)
        output_clean = test_model(poison_x,poison_edge_index,poison_edge_weights)
        ori_predict = torch.exp(output_clean[known_nodes])
        print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))
        print(f"Running for drop_ratio={drop_ratio}, K={K}")

        # 重新定义 sample_noise_all，传入当前 drop_ratio
        def sample_noise_all(edge_index, edge_weight, device, drop_ratio=drop_ratio):
            edge_index = edge_index.to(device)
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=device)
            else:
                edge_weight = edge_weight.to(device)

            drop_mask = Bernoulli(1 - drop_ratio).sample(edge_weight.size()).bool()
            noisy_edge_index = edge_index[:, drop_mask]
            noisy_edge_weight = edge_weight[drop_mask]

            node_degrees = torch.zeros(edge_index.max() + 1, device=device)
            node_degrees.index_add_(0, noisy_edge_index[0], torch.ones(noisy_edge_index.size(1), device=device))

            isolated_nodes = node_degrees == 0
            if isolated_nodes.any():
                potential_restore_edges = isolated_nodes[edge_index[0]]
                restore_edges = edge_index[:, potential_restore_edges]
                noisy_edge_index = torch.cat([noisy_edge_index, restore_edges], dim=1)
                restored_weights = torch.ones(restore_edges.size(1), device=device)
                noisy_edge_weight = torch.cat([noisy_edge_weight, restored_weights], dim=0)

            return noisy_edge_index, noisy_edge_weight
        
        def find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, target_class):
            # Get the specific list to iterate through
            labels_list = poison_labels[bkd_tn_nodes[index_of_less_robust]]

            # Iterate through the list with index
            for i in range(len(labels_list) - 1):  # -1 to avoid index out of range
                if labels_list[i] != target_class and labels_list[i + 1] != target_class:
                    return i - 1
            
            return None
        # 清空历史predictions
        predictions = []
        
        # 用当前K跑
        for i in range(K):
            test_model.eval()
            noisy_poison_edge_index, noisy_poison_edge_weights = sample_noise_all(poison_edge_index, poison_edge_weights, device)
            output = test_model(poison_x, noisy_poison_edge_index, noisy_poison_edge_weights)
            predictions.append(torch.exp(output[known_nodes]))

        epsilon = 1e-8
        deviations = []
        for sub_pred in predictions:
            sub_pred += epsilon
            deviation = F.kl_div(sub_pred.log(), ori_predict, reduce=False)
            deviations.append(deviation)
        
    

        summed_deviations = torch.zeros_like(deviations[0]).to(deviations[0].device)
        for deviation in deviations:
            summed_deviations += deviation
        
        PV = torch.mean(summed_deviations, dim=-1)
        PV = PV.detach().cpu().numpy()
        plt.figure(figsize=(7,5))

        plt.hist(PV[:-len(idx_attach)], bins=30, color='blue', alpha=0.5, label="干净节点", edgecolor='black')

        # 受感染节点：crimson
        plt.hist(PV[-len(idx_attach):], bins=30, color='crimson', alpha=0.7, label="受感染节点", edgecolor='black')

        plt.xlabel("删边前后预测值差异", fontsize=20)
        plt.ylabel("节点数量", fontsize=20)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        index_of_less_robust = torch.sort(torch.mean(summed_deviations, dim=-1), descending=True)[1]
        result_index = find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, args.target_class)

        idx_poison_found = known_nodes[index_of_less_robust][:result_index]
        idx_clean_found = known_nodes[index_of_less_robust[result_index:]]

        poison_common = set(idx_poison_found.tolist()) & set(idx_attach.tolist())
        clean_common = set(idx_clean_found.tolist()) & set(idx_train.tolist())


        TP = len(poison_common)   # 正确检测出的中毒节点
        FP = len(idx_poison_found) - TP  # 被误判为中毒节点的干净节点
        FN = len(idx_attach) - TP  # 没有被检测出的中毒节点
        TN = len(clean_common)  # 正确识别的干净节点

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        fpr = FP / (FP + TN + 1e-8)

        ####### 训练防御后模型 #######
        test_model = model_construct(args, args.test_model, data, device).to(device)
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=400, verbose=False, finetune1=True, attach=known_nodes[index_of_less_robust][:result_index])

        induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights, torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)])

        clean_acc = test_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

        induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(idx_atk, poison_x, induct_edge_index, induct_edge_weights, device)
        induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()

        output = test_model(induct_x, induct_edge_index, induct_edge_weights)

        asr = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean().item()
        flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
        flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
        ca = test_model.test(induct_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

        results[(drop_ratio, K)] = {
            'ASR': flip_asr,
            'CleanACC': clean_acc,
            'Precision': precision,
            'Recall': recall,
            'FPR': fpr,
        }

        for key, value in results.items():
            print(f"drop_ratio={key[0]}, K={key[1]} -> "
                f"ASR: {value['ASR']:.6f}, "
                f"CleanAcc: {value['CleanACC']:.6f}, "
                f"Precision: {value['Precision']:.6f}, "
                f"Recall: {value['Recall']:.6f}, "
                f"FPR: {value['FPR']:.6f}")
