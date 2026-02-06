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

        #RIGBD Finetune
    def finetune1(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose):
        
        idx_train_set = set(idx_train)
        idx_attach_set = set(idx_attach)
        idx1 = list(idx_train_set - idx_attach_set)
        idx2 = idx_attach

        idx1 = torch.tensor(idx1)
        idx2 = torch.tensor(idx2)
        

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, x = self.forward(self.features, self.edge_index, self.edge_weight)
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
    def finetune2(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, target_label, gamma):     

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        num_classes=labels.max().item() + 1
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[idx_attach] = 1  
        choices = torch.tensor([i for i in range(num_classes) if i != target_label])
        random_replacements = choices[torch.randint(0, len(choices), size=(mask.sum(),))]
        reshuffle_labels = labels
        reshuffle_labels[mask] = random_replacements  
        for i in range(train_iters): 
            self.train()
            optimizer.zero_grad()
            output, x = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_attach_forget = -1 * F.nll_loss(output[idx_attach], labels[idx_attach], reduction='none')
            loss_attach_forget = torch.relu(loss_attach_forget)
            loss_attach_decouple =  F.nll_loss(output[idx_attach], reshuffle_labels[idx_attach], reduction='none')
            loss_clean = F.nll_loss(output[idx_clean], labels[idx_clean], reduction='none')

            loss_attach = gamma * loss_attach_decouple + (1-gamma) * loss_attach_forget

            loss_train = torch.cat([loss_clean, loss_attach])
            loss_train = torch.mean(loss_train)
            loss_train.backward()
            optimizer.step()

        self.eval()
        self.output = output
    
    #Discarding Target Nodes/Only use Clean nodes 
    def finetune3(self, labels, idx_train, idx_val, idx_clean, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, x = self.forward(self.features, self.edge_index, self.edge_weight)

            loss_train = F.nll_loss(output[idx_clean], labels[idx_clean])
            loss_train.backward()
            optimizer.step()
        self.eval()
        self.output = output
    
    #SCRUB Unlearning
    def finetune4(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, target_label, teacher_model):
        """
        SCRUB unlearning
        """
        with torch.no_grad():
            teacher_model.eval()
            teacher_output, x = teacher_model.forward(self.features, self.edge_index, self.edge_weight)
            teacher_probs_clean = F.softmax(teacher_output[idx_clean], dim=1)
            teacher_probs_attach = F.softmax(teacher_output[idx_attach], dim=1)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for epoch in range(10):
            for step in range(1): 
                self.train()
                optimizer.zero_grad()
                
                student_output, x = self.forward(self.features, self.edge_index, self.edge_weight)
                student_probs_attach = F.log_softmax(student_output[idx_attach], dim=1)
                
                max_loss = -F.kl_div(student_probs_attach, teacher_probs_attach, reduction='batchmean')
                
                max_loss.backward()
                optimizer.step()
            
            for step in range(200): 
                self.train()
                optimizer.zero_grad()
                
                student_output, x = self.forward(self.features, self.edge_index, self.edge_weight)
                student_probs_clean = F.log_softmax(student_output[idx_clean], dim=1)

                kl_clean = F.kl_div(student_probs_clean, teacher_probs_clean, reduction='batchmean')
                task_loss = F.nll_loss(student_output[idx_clean], labels[idx_clean])
                
                min_loss = 0.5*kl_clean + 0.5*task_loss
                
                min_loss.backward()
                optimizer.step()
        
        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    #Restore Original Label
    def finetune5(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, clean_labels):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters): 
            self.train()
            optimizer.zero_grad()
            output, x = self.forward(self.features, self.edge_index, self.edge_weight)

            loss_train = F.nll_loss(output[idx_train], clean_labels[idx_train])
            loss_train.backward()
            optimizer.step()

        self.eval()
        self.output = output
    
    #Feature Reinitialization
    def finetune6(self, labels, idx_train, idx_attach, idx_val, train_iters, verbose, attach_feature):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters): 
            self.train()
            optimizer.zero_grad()
            output, x= self.forward(attach_feature, self.edge_index, self.edge_weight)


            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

        self.eval()
        self.output = output

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, attach=None, clean=None, finetune1=False, finetune2=False, finetune3=False, finetune4=False, finetune5=False, finetune6=False, target_label=0, gamma=0.7, teacher_model=None, clean_labels=None, attach_feature=None):
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
                self.finetune2(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label, gamma)
            elif finetune3 == True:
                self.finetune3(self.labels, idx_train, idx_val, clean, train_iters, verbose)
            elif finetune4 == True:
                self.finetune4(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label, teacher_model)
            elif finetune5 == True:
                self.finetune5(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, clean_labels)
            elif finetune6 == True:
                self.finetune6(self.labels, idx_train, idx_val, attach, train_iters, verbose, attach_feature)
            else:
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

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
