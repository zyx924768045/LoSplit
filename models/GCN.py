#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from torch_geometric.utils import from_scipy_sparse_matrix
import torch.nn as nn
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, norm, wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter


class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""

    def __init__(self, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=1e-20, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-20, max=1)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss



class GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False,add_self_loops=True):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.add_self_loops = add_self_loops
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, add_self_loops=self.add_self_loops))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid, add_self_loops=self.add_self_loops))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass, add_self_loops=self.add_self_loops)
        # print('add_selfloop',self.gc2.add_self_loops)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.activations = []
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.rce = RCELoss(num_classes = self.nclass, reduction='none')

    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        final_pre_act = self.gc2(x, edge_index, edge_weight)
        # self.pre_activations.append(final_pre_act)
        return F.log_softmax(final_pre_act,dim=1)
    
    def get_h(self, x, edge_index):
        if x.shape[1] > 1024 or x.shape[1] < 256:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                mean = x.mean(dim=1, keepdim=True) 
                std = x.std(dim=1, keepdim=True)  
                std = torch.where(std == 0, torch.ones_like(std), std)
                x = (x - mean) / std
        else:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
        return x


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune1=False, finetune2=False, finetune3=False, finetune4=False, attach=None, clean=None, target_label=0, num_attach=40, alpha=0.7):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            if finetune1==True:
                self.finetune1(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose)
            elif finetune2 == True:
                self.finetune2(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label, alpha)
            elif finetune3 == True:
                self.finetune3(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label)
            else:
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose, num_attach)
        # torch.cuda.empty_cache()

    #RIGBD Finetune
    def finetune1(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose):
        idx_train_set = set(idx_train)
        idx_attach_set = set(idx_attach)
        idx1 = list(idx_train_set - idx_attach_set)
        idx2 = idx_attach

        idx1 = torch.tensor(idx1).to(self.device)
        idx2 = torch.tensor(idx2).to(self.device)
        

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx1], labels[idx1])
            probs = F.softmax(output[idx2], dim=1)
            target_probs = probs[range(len(labels[idx2])), labels[idx2]]
            loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

            # Combining the normal and adversarial losses
            loss_train = loss_train + loss_train_2
            loss_train.backward()
            optimizer.step()
        self.eval()
        self.output = output
    
    #LoSplit Finetune
    def finetune2(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, target_label, alpha):
        num_classes=labels.max().item() + 1
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[idx_attach] = 1  
        choices = torch.tensor([i for i in range(num_classes) if i != target_label])
        random_replacements = choices[torch.randint(0, len(choices), size=(mask.sum(),))]
        reshuffle_labels = labels
        reshuffle_labels[mask] = random_replacements        

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_attach_forget = -1 * F.nll_loss(output[idx_attach], labels[idx_attach], reduction='none')
            loss_attach_forget = torch.relu(loss_attach_forget)
            loss_attach_decouple =  F.nll_loss(output[idx_attach], reshuffle_labels[idx_attach], reduction='none')
            loss_clean = F.nll_loss(output[idx_clean], labels[idx_clean], reduction='none')

            loss_attach = alpha * loss_attach_decouple + (1-alpha) * loss_attach_forget

            loss_train = torch.cat([loss_clean, loss_attach])
            loss_train = torch.mean(loss_train)
            loss_train.backward()
            optimizer.step()

        self.eval()
        self.output = output
    
    #Only use Clean nodes 
    def finetune3(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, target_label):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)

            loss_train = F.nll_loss(output[idx_clean], labels[idx_clean], reduction='none')
            # loss_train = self.sce(output[idx_train], labels[idx_train])
            loss_train = torch.mean(loss_train)
            loss_train.backward()
            optimizer.step()
        self.eval()
        self.output = output
    
    
    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
        
        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose, num_attach):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], reduction = 'none')
            # ce = torch.nn.CrossEntropyLoss(reduction='none')
            # loss_train = self.sce(output[idx_train], labels[idx_train])
            # loss_train = ce(output[idx_train], labels[idx_train])
            loss_train = torch.mean(loss_train)
            # loss_clean = torch.mean(loss_train[:-31])
            # loss_clean.backward()
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        # plt.show()

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # torch.cuda.empty_cache()


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids




    

# %%
