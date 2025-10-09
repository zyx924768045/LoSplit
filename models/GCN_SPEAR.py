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
import torch.nn as nn
import matplotlib.pyplot as plt

clean_loss_list = []
poisoned_loss_list = []
train_loss_list = []

class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""

    def __init__(self, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=1e-7, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class SCELoss(nn.Module):
    """Symmetric Cross Entropy."""

    def __init__(self, alpha=0.1, beta=1, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction)
        ce_loss = ce(x, target)
        rce_loss = rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss

        return loss

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False,add_self_loops=True):

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

    def forward(self, x, edge_index, edge_weight=None):
        self.pre_activations = []
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            pre_act = conv(x, edge_index, edge_weight)  # 线性变换结果
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            self.activations.append(x)
            x = F.dropout(x, self.dropout, training=self.training)
            self.pre_activations.append(pre_act)
        # x = self.gc2(x, edge_index,edge_weight)
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

    def compute_loss_value(self,idx_train, labels):
        criterion = SCELoss(alpha=0.1, beta=1, num_classes=labels.max().item() + 1, reduction="none")
        # criterion = nn.NLLLoss(reduction = 'none').to(self.device)
        # criterion = nn.CrossEntropyLoss(reduction='none')
        self.eval()
        with torch.no_grad():
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            losses_record = criterion(output[idx_train],self.labels[idx_train])

        # plt.figure(figsize=(8, 5))
        # plt.hist(losses_record[:-551], bins=20, alpha=0.6, color='blue', density=False, label='Clean Nodes Loss')
        # plt.hist(losses_record[-551:], bins=20, alpha=0.6, color='red', density=False, label='Poisoned Nodes Loss')

        # plt.xlabel("Loss Value")
        # plt.ylabel("Number of Samples")  # 改为样本数量
        # plt.title("Distribution of Clean/Poisoned SCE Loss")
        # plt.legend()
        # plt.grid(True)

        # plt.show()
        idx_losses_record = np.argsort(np.array(losses_record.detach().cpu()))
        
        # print(losses_record[-5:], losses_record[:-5])
        return idx_losses_record
    
    def isolate_data(self, idx_train, labels, isolation_ratio, clean_ratio):
        idx_losses_record = self.compute_loss_value(idx_train, labels)
        idx_clean = idx_train[idx_losses_record[int(len(idx_losses_record)* clean_ratio):]]
        idx_isolated = idx_train[idx_losses_record[0:int(len(idx_losses_record)* isolation_ratio)]]

        return idx_isolated, idx_clean

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune=False, attach=None, clean=None, loss_increse_threshold=1, lossplit=False, isolate_epoch=20, isolation_ratio=0.06, clean_ratio=0.07, target_label=0):
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
            if finetune==True:
                self.finetune(self.labels, idx_train, idx_val, attach, clean, train_iters, verbose, loss_increse_threshold, target_label)
            elif lossplit==True:
                idx_isolated, idx_clean = self.lossplit(self.labels, idx_train, verbose, isolate_epoch, isolation_ratio, clean_ratio)
                return idx_isolated, idx_clean
            else:
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        # torch.cuda.empty_cache()

    def finetune(self, labels, idx_train, idx_val, idx_attach, idx_clean, train_iters, verbose, loss_increse_threshold, target_label):
        # idx1 = idx_train[:-len(idx_attach)]
        # idx2 = idx_train[-len(idx_attach):]
        # idx1 = [item for item in idx_train if item not in idx_attach]
        num_classes=labels.max().item() + 1
        new_labels = torch.randint(1, num_classes, labels.shape)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[idx_attach] = 1  
        labels = torch.where(mask, new_labels, labels)

        idx_train_set = set(idx_clean)
        idx_attach_set = set(idx_attach)
        idx1 = list(idx_train_set - idx_attach_set)
        idx2 = idx_attach

        idx1 = torch.tensor(idx1).to(self.device)
        idx2 = torch.tensor(idx2).to(self.device)
        

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_sce = SCELoss(alpha=0.1, beta=1, num_classes=labels.max().item() + 1, reduction="none")
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train_isolated = -1 * loss_sce(output[idx_attach], labels[idx_attach])
            # # loss_train_isolated = -1 * F.nll_loss(output[idx_attach], labels[idx_attach])
            # loss_train_isolated = torch.relu(loss_train_isolated + loss_increse_threshold)
            # loss_train_isolated = torch.mean(loss_train_isolated)
            # # loss_train_isolated.backward()


            # # probs = F.softmax(output[idx_attach], dim=1)
            # # target_probs = probs[range(len(labels[idx_attach])), labels[idx_attach]]
            # # loss_train_isolated = torch.mean(target_probs)  # Mean of probabilities of correct labels
            # # loss_train_isolated.backward()
            # print('Epoch {}, loss_train_isolated: {}'.format(i, loss_train_isolated.item()))
            # # optimizer.step()

            # # optimizer.zero_grad()
            # # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # # loss_train_clean = F.nll_loss(output[idx_clean], labels[idx_clean])
            # loss_train_clean = loss_sce(output[idx_clean], labels[idx_clean])
            # loss_train_clean = torch.mean(loss_train_clean)
            # loss_train = loss_train_clean + loss_train_isolated
            # loss_train = torch.mean(loss_train)
            # loss_train.backward()
            # # loss_train_clean.backward()
            # print('Epoch {}, loss_train_clean: {}'.format(i, loss_train_clean.item()))
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], reduction='none')
            # loss_train = loss_sce(output[idx_train], labels[idx_train])
            loss_train = torch.mean(loss_train)
            loss_train.backward()
            print('Epoch {}, loss_train_isolated: {}'.format(i, loss_train.item()))
            optimizer.step()
        self.eval()
        self.output = output

    def lossplit(self, labels, idx_train, verbose, isolate_epoch, isolation_ratio, clean_ratio):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sce = SCELoss(alpha=0.1, beta=1, num_classes=labels.max().item() + 1, reduction="none")
        # criterion = nn.NLLLoss(reduction = 'none').to(self.device)
        # ce = nn.CrossEntropyLoss(reduction='none')
        found = []
        for i in range(isolate_epoch):
            self.train()
            # threshold = self.threshold
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train], reduction='none')
            loss_train = sce(output[idx_train], labels[idx_train])
            idx_isolated, idx_clean = self.isolate_data(idx_train, labels, isolation_ratio, clean_ratio)
            # loss_train = torch.sign(loss_train - threshold) * loss_train
            poison_common = set(idx_isolated.tolist()) & set(idx_train[-551:].tolist())
            clean_common = set(idx_clean.tolist()) & set(idx_train[:-551].tolist())
            found.append(len(poison_common)+len(clean_common))
            print("total_isolated:", len(idx_isolated))
            print("total_clean:", len(idx_clean))
            print("poisoned nodes found:", len(poison_common))
            print("clean nodes found:", len(clean_common))
            print("--------------------------------")
            loss_train = torch.mean(loss_train)
            loss_train.backward()
            optimizer.step()
        best_isolate_epoch = np.argmax(found)
        print("best_isolate_epoch:", best_isolate_epoch)

        # for i in range(best_isolate_epoch):
        #     self.train()
        #     optimizer.zero_grad()
        #     output = self.forward(self.features, self.edge_index, self.edge_weight)
        #     # loss_train = F.nll_loss(output[idx_train], labels[idx_train], reduction='none')
        #     loss_train = sce(output[idx_train], labels[idx_train])
        #     idx_isolated, idx_clean = self.isolate_data(idx_train, labels, isolation_ratio, clean_ratio)
        #     poison_common = set(idx_isolated.tolist()) & set(idx_train[-159:].tolist())
        #     clean_common = set(idx_clean.tolist()) & set(idx_train[:-159].tolist())

        #     # print("total_isolated:", len(idx_isolated))
        #     # print("total_clean:", len(idx_clean))
        #     # print("poisoned nodes found:", len(poison_common))
        #     # print("clean nodes found:", len(clean_common))

        #     loss_train = torch.mean(loss_train)
        #     loss_train.backward()
        #     optimizer.step()
        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        return idx_isolated, idx_clean

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
        loss_sce = SCELoss(alpha=0.1, beta=1, num_classes=labels.max().item() + 1, reduction="none")
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train], reduction = 'none')
            # ce = torch.nn.CrossEntropyLoss(reduction='none')
            loss_train = loss_sce(output[idx_train], labels[idx_train])
            # loss_train = ce(output[idx_train], labels[idx_train])
            clean_loss_list.append(torch.mean(loss_train[:-551]).item())
            poisoned_loss_list.append(torch.mean(loss_train[-551:]).item())
            train_loss_list.append(torch.mean(loss_train).item())
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
        plt.figure(figsize=(8, 5))
        plt.plot(clean_loss_list, label="Clean Nodes Loss", color='blue')
        plt.plot(poisoned_loss_list, label="Poisoned Nodes Loss", color='red')
        plt.plot(train_loss_list, label="bkd_td_nodes Loss", color='black')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve (Clean vs Poisoned Nodes vs bkd_tn_nodes)")
        plt.legend()
        plt.grid()
        plt.show()

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
