import argparse
import numpy as np
import torch
from copy import deepcopy

from torch_geometric.datasets import Planetoid,Flickr,Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from help_funcs import reconstruct_prune_unrelated_edge

# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge
import scipy.sparse as sp
from select_sample import select
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import norm
import pandas as pd


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
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv','Computers','Photo', 'Citeseer', 'Physics'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)

parser.add_argument('--target_class', type=int, default=0)

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--rec_epochs', type=int,  default=100,
                    help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')

#Backdoor Setting
parser.add_argument('--shadow_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trojan_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=50,
                    help="number of poisoning nodes relative to the full graph")
parser.add_argument('--target_loss_weight', type=float, default=2,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=1.2,
                    help="Weight of optimize similarity loss")


#Attack Setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GNNGuard','RobustGCN', 'ABL'],
                    help='Model used to attack')

#OD&Prune Setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'none','reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.1,
                    help="Threshold of prunning edges")


#ABL Setting
parser.add_argument('--isolation_ratio', type=float, default=0.01,
                    help="ABL isolation ratio")
parser.add_argument('--isolate_epoch', type=int, default=20,
                    help="ABL isolate epoch")
parser.add_argument('--loss_increse_threshold', type=float, default=1.4,
                    help="ABL loss_increse_threshold") 
parser.add_argument('--threshold', type=float, default=0.5,
                    help="ABL threshold") 

#LoSplit Setting                    
parser.add_argument('--split_lr', type=float, default=0.05,
                    help='split learning rate.')
parser.add_argument('--split_epoch', type=int, default=10,
                    help="split target nodes")
parser.add_argument('--gamma', type=float, default=0.7,
                    help='Decoupling trade off')

# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")

#SPEAR Setting
parser.add_argument('--alpha', type=float, default=0.1,
                    help="Ratio of feature dimensions to perturb")
parser.add_argument('--alpha_int', type=int, default=4,
                    help="Number of feature dimensions to perturb")
parser.add_argument('--outter_size', type=int, default=512,
                    help="Number of outter samples")

#Pre-train Setting
parser.add_argument('--trigger_generator_address', type=str, default='./weights/SPEAR/Pubmed/SPEAR_Pubmed_weights.pth')
parser.add_argument('--pre_train_param', type=str, default='./weights/SPEAR/Pubmed/SPEAR_Pubmed.pt')

#Attack Type
parser.add_argument('--attack', type=str, default='SPEAR',
                    help="attack type")


args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
print(args)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
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
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 
elif(args.dataset == 'Physics'):
    from torch_geometric.datasets import Coauthor
    dataset = Coauthor(root='./data/Physics', name='Physics')


data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
elif(args.dataset=='Physics'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


from models.backdoor_SPEAR import Backdoor
from models.construct import model_construct

unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("#Attach Nodes:{}".format(size))
assert size>0, 'The number of selected trigger nodes must be larger than 0!'
idx_attach = select(data,args,idx_train,idx_val,device).to(device)

idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
print("idx_attach: {}".format(idx_attach))
print("num_attach: ", len(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print(unlabeled_idx)
model = Backdoor(args,device)
# model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
# torch.save(model.trojan.state_dict(), args.trigger_generator_address)

model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx, args.trigger_generator_address, True)

poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()


if(args.defense_mode == 'prune'):
    poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'reconstruct'):
    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,data.x,data.edge_index,device, idx_attach, large_graph=True)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))

known_nodes = torch.cat([idx_train,idx_attach]).to(device)
# edge weight for clean edge_index, may use later #
edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)


# data_loaded = torch.load(args.pre_train_param, map_location='cpu')
# idx_poison_found = data_loaded['idx_poison_found']
# idx_clean_found = data_loaded['idx_clean_found']
# target_label = data_loaded['target_label']
# poison_x = data_loaded['poison_x']
# poison_edge_index = data_loaded['poison_edge_index']
# poison_edge_weights = data_loaded['poison_edge_weights']
# poison_labels = data_loaded['poison_labels']

test_model = model_construct(args,args.test_model,data,device).to(device) 
if args.test_model == 'ABL':
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, num_attach=len(idx_attach))
else:
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
    # test_model.fit(poison_x, poison_edge_index, poison_edge_weights, data.y, bkd_tn_nodes, idx_val,train_iters=200,verbose=False, finetune5=True)
    # test_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)



output = test_model(poison_x,poison_edge_index,poison_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
if(args.defense_mode == 'prune'):
    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
output = test_model(induct_x,induct_edge_index,induct_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
# print("ASR: {:.6f}".format(train_attach_rate))
asr = train_attach_rate
flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
output = output.cpu()

print("****After SPEAR Attack****")
print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
print("Clean Accuracy: {:.6f}".format(clean_acc)) 
test_model = test_model.cpu()



#LoSplit Defense###
LoSplit= model_construct(args,'LoSplit',data,device,split_lr=args.split_lr).to(device) 
poison_found, clean_found, deltas, target_labels = LoSplit.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, known_nodes, idx_val,train_iters=args.epochs,verbose=False, split_epoch=args.split_epoch, num_attach=len(idx_attach))

best_delta = max(deltas[1:])  
best_t = [i for i, val in enumerate(deltas) if val == best_delta][0] 
idx_poison_found = poison_found[best_t]
idx_clean_found = clean_found[best_t]
target_label = target_labels[best_t]

data_to_save = {
    'idx_poison_found': idx_poison_found,
    'idx_clean_found': idx_clean_found,
    'target_label': target_label,
    'poison_x': poison_x,
    'poison_edge_index': poison_edge_index,
    'poison_edge_weights': poison_edge_weights,
    'poison_labels': poison_labels
}


torch.save(data_to_save, args.pre_train_param)


test_model = model_construct(args, args.test_model, data, device).to(device)
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=200, verbose=False, finetune2=True, attach=idx_poison_found, clean=idx_clean_found,target_label=target_label, alpha=args.gamma)

induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
induct_edge_weights = torch.cat([
    poison_edge_weights,
    torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)
])
clean_acc = test_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(
    idx_atk, poison_x, induct_edge_index, induct_edge_weights, device
)
induct_x, induct_edge_index, induct_edge_weights = (
    induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
)

output = test_model(induct_x, induct_edge_index, induct_edge_weights)
flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
flip_asr = (output.argmax(dim=1)[flip_idx_atk] == args.target_class).detach().cpu().numpy().mean()

print("****After LoSplit Defense****")
print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
print("Clean Accuracy: {:.6f}".format(clean_acc)) 

poison_common = set(idx_poison_found.tolist()) & set(idx_attach.tolist())
clean_common = set(idx_clean_found.tolist()) & set(idx_train.tolist())

TP = len(poison_common)   
FP = len(idx_poison_found) - TP 
FN = len(idx_attach) - TP 
TN = len(clean_common) 

precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
fpr = FP / (FP + TN + 1e-8)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"FPR: {fpr:.4f}")

