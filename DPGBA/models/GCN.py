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
from torch_geometric.utils import from_scipy_sparse_matrix


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False, add_self_loops=True):

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
        self.ce = nn.CrossEntropyLoss(reduction='none')

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
        features = x
        # print('features',features)
        x = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(x,dim=1), features
    
    def get_h(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

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
    
    #Only finetune clean nodes
    def finetune3(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, target_label):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, x = self.forward(self.features, self.edge_index, self.edge_weight)

            loss_train = F.nll_loss(output[idx_clean], labels[idx_clean], reduction='none')
            loss_train = torch.mean(loss_train)
            loss_train.backward()
            optimizer.step()
        self.eval()
        self.output = output
    
    #Finetune with SCRUB Unlearning
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
    

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, attach=None, clean=None, finetune1=False, finetune2=False, finetune3=False, finetune4=False, target_label=0, gamma=0.7, teacher_model=None):
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
                self.finetune2(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label, gamma)
            elif finetune3 == True:
                self.finetune3(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label)
            elif finetune4 == True:
                self.finetune4(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, target_label, teacher_model)
            else:
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        # torch.cuda.empty_cache()


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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, x = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output, x = self.forward(self.features, self.edge_index, self.edge_weight)
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
            output, x = self.forward(features, edge_index, edge_weight)
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
