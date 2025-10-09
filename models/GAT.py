#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8,dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True ,device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GATConv(nfeat,nhid,heads,dropout=dropout)
        self.gc2 = GATConv(heads*nhid, nclass, concat=False, dropout=dropout)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def forward(self, x, edge_index, edge_weight=None): 
        x = F.dropout(x, p=self.dropout, training=self.training)    # optional
        # x = F.elu(self.gc1(x, edge_index, edge_weight))   # may apply later 
        x = F.elu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index, edge_weight)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x,dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
    def get_h(self, x, edge_index):
        """
        Return hidden representation before the final classification layer.
        """
        x = F.elu(self.gc1(x, edge_index))  # shape: [N, heads * nhid]
        
        # Optional: normalize if dim is too large or too small
        if x.shape[1] > 1024 or x.shape[1] < 256:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            x = (x - mean) / std
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, finetune1=False, finetune2=False, finetune3=False, finetune4=False, attach=None, clean=None, target_label=0, num_attach=0):
        if initialize:
            self.initialize()
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)


        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            if finetune1==True:
                self.finetune1(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose)
            elif finetune2 == True:
                self.finetune2(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label)
            elif finetune3 == True:
                self.finetune3(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label)
            elif finetune4 == True:
                self.finetune4(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label)
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

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
    def finetune2(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, target_label):
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
            loss_attach_unlearn = -1 * F.nll_loss(output[idx_attach], labels[idx_attach], reduction='none')
            loss_attach_unlearn = torch.relu(loss_attach_unlearn)
            loss_attach_decouple =  F.nll_loss(output[idx_attach], reshuffle_labels[idx_attach], reduction='none')
            loss_clean = F.nll_loss(output[idx_clean], labels[idx_clean], reduction='none')
            loss_attach = loss_attach_decouple + loss_attach_unlearn

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
    
    #Without Unlearning
    def finetune4(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, target_label):
        num_classes=labels.max().item() + 1
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[idx_attach] = 1  
        choices = torch.tensor([i for i in range(num_classes) if i != target_label])
        random_replacements = choices[torch.randint(0, len(choices), size=(mask.sum(),))]
        labels[mask] = random_replacements        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], reduction='none')
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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index,self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids
# %%
