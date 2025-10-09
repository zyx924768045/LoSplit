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
from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = 'Helvetica'  # 或 'Arial', 'DejaVu Sans'
# # plt.rcParams['font.size'] = 11
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
# plt.rcParams['savefig.bbox'] = 'tight'
# plt.rcParams['legend.fontsize'] = 9
# plt.rcParams['xtick.labelsize'] = 16
# plt.rcParams['ytick.labelsize'] = 16


class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""

    def __init__(self, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=1e-10, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-10, max=1.0)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class LoSplit(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False,add_self_loops=True):

        super(LoSplit, self).__init__()

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
        self.rce = RCELoss(num_classes = self.nclass, reduction='none')
        self.args = args

    def forward(self, x, edge_index, edge_weight=None):
        self.pre_activations = []
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index,edge_weight)
        final_pre_act = self.gc2(x, edge_index, edge_weight)
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
        self.eval()
        with torch.no_grad():
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            losses_record = self.rce(output[idx_train],self.labels[idx_train])

        idx_losses_record = np.argsort(np.array(losses_record.detach().cpu()))
    
        losses_np = losses_record.detach().cpu().numpy()
        losses_np.sort()

        return idx_losses_record, losses_np

    def split_node(self, idx_train, labels, num_epoch, num_attach):
        idx_losses_record, losses_np = self.compute_loss_value(idx_train, labels)
        labels_list = labels[idx_train[idx_losses_record]]
        labels_np = np.array(labels_list)
        

        loss_features = np.tile(losses_np.reshape(-1, 1), (1, 5))  
        loss_features += np.random.normal(0, 0.01, size=loss_features.shape)  

        classes = np.unique(labels)

        class_vars = [np.var(losses_np[labels_list == c]) for c in classes]
        target_label = classes[np.argmax(class_vars)]
        # print(f"Target label (max variance): {target_label}")
    
        target_mask = labels_np == target_label
        target_indices = np.where(target_mask)[0]
        target_losses = losses_np[target_mask]
        target_losses_reshaped = target_losses.reshape(-1, 1)

        # Fit GMM
        gmm = GaussianMixture(n_components=2, random_state=42).fit(target_losses_reshaped)
        cluster_labels = gmm.predict(target_losses_reshaped)

        means = gmm.means_.flatten()

        # Identify low-loss cluster (target)
        poison_cluster = np.argmin(means)
        clean_cluster = np.argmax(means)

        poison_losses = target_losses[cluster_labels == poison_cluster]
        clean_losses = target_losses[cluster_labels == clean_cluster]
        poison_mean = np.mean(poison_losses)
        clean_mean = np.mean(clean_losses)
        delta = clean_mean - poison_mean

        #Compute z-score
        mean = target_losses.mean()
        std = target_losses.std()  + 1e-8
        z_scores = (target_losses - mean) / (std)
        
        # Set Threshold
        target_cluster_zscores = z_scores[cluster_labels == poison_cluster]
        clean_cluster_zscores = z_scores[cluster_labels == clean_cluster]

        

        if len(poison_losses) == 0 or len(clean_losses) == 0:
            threshould = 1e-10
        else:
            threshould = np.max(poison_losses)+  (np.min(clean_losses) - np.max(poison_losses)) / 2

        if len(target_cluster_zscores) == 0 or len(clean_cluster_zscores) == 0:
            z_thresh = 1e-3
        else:
            z_thresh = np.max(clean_cluster_zscores) + (np.min(target_cluster_zscores) - np.max(clean_cluster_zscores)) / 2


        # selected_target_mask = z_scores < z_thresh
        selected_target_mask = target_losses <= threshould
        suspected_target_indices = target_indices[selected_target_mask]
        split_point = len(suspected_target_indices)

        target_indices = np.where(target_mask)[0] 

        poison_indices = target_indices[:split_point]
        poison_mask = np.zeros_like(target_mask, dtype=bool)
        poison_mask[poison_indices] = True
        clean_mask = ~poison_mask

        idx_poison_found = idx_train[idx_losses_record[poison_mask]]
        idx_clean_found = idx_train[idx_losses_record[clean_mask]]

        # idx_attach = idx_train[-num_attach:]
        # poison_common = set(idx_poison_found.tolist()) & set(idx_attach.tolist())
        # clean_common = set(idx_clean_found.tolist()) & set(idx_train.tolist())

        # TP = len(poison_common)
        # FP = len(idx_poison_found) - TP
        # FN = len(idx_attach) - TP
        # TN = len(clean_common)

        # precision = TP / (TP + FP + 1e-8)
        # recall = TP / (TP + FN + 1e-8)
        # fpr = FP / (FP + TN + 1e-8)

        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"FPR: {fpr:.4f}")
        # idx_filtered = idx_train[idx_losses_record] 

        # mask_poison = torch.isin(idx_filtered, idx_attach)
        # mask_clean = ~mask_poison

        # clean_labels_list = labels[idx_filtered[mask_clean]]
        # labels_list = labels[idx_train[idx_losses_record]]
        
        # mask_clean_np = mask_clean.detach().cpu().numpy()
        # mask_poison_np = mask_poison.detach().cpu().numpy()
        # losses_clean_np = losses_np[mask_clean_np]
        # losses_poison_np = losses_np[mask_poison_np]


        # cmap = plt.get_cmap('tab10')
        # unique_classes = np.unique(labels)

        # fig_width = 1.75
        # fig_height = 1.62
        # plt.figure(figsize=(fig_width, fig_height))

        # font_scale = 1.0
        # label_size = 10 * font_scale
        # title_size = 8 * font_scale
        # legend_size = 3 * font_scale
        # tick_size = 9 * font_scale

        # line_scale = 1.0
        # hist_linewidth = 0.5 * line_scale
        # grid_linewidth = 0.6 * line_scale
        # spine_linewidth = 1.0 * line_scale

        # for i, cls in enumerate(unique_classes):
        #     if cls == self.args.target_class:
        #         cls_mask = (clean_labels_list == cls)
        #         plt.hist(losses_clean_np[cls_mask], bins=20, alpha=0.75,
        #                 color=cmap(i % 10), edgecolor='black',
        #                 linewidth=hist_linewidth, 
        #                 label=f'Clean (Class {self.args.target_class})')

        # plt.hist(losses_poison_np, bins=30, alpha=0.75, color='crimson',
        #         edgecolor='black', linewidth=hist_linewidth,
        #         hatch='//', label='Target Nodes')

        # plt.axvline(x=threshould, color='red', linestyle='--',
        #             linewidth=1.2 * line_scale, label='Threshold')

        # plt.xlabel("Loss Value", fontsize=label_size, fontweight='bold')
        # plt.ylabel("Node Count", fontsize=label_size, fontweight='bold')
        # plt.title(f"Loss Distribution (Epoch {num_epoch})", fontsize=title_size, fontweight='bold')

        # # plt.legend(fontsize=legend_size, loc='upper left', frameon=False, prop={'weight':'bold'})
        # plt.legend(
        #         fontsize=0.5,         
        #         loc='upper left', 
        #         frameon=False, 
        #         prop={'weight': 'bold'},
        #         handlelength=1.25,            
        #         handletextpad=0.3,           
        #         markerscale=0.6               
        #     )

        # ax = plt.gca()
        # ax.tick_params(axis='both', labelsize=tick_size, length=2.5, width=0.8)

        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label1.set_fontweight('bold')
        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label1.set_fontweight('bold')

        # for spine in ax.spines.values():
        #     spine.set_linewidth(spine_linewidth)

        # plt.grid(True, linestyle='--', linewidth=grid_linewidth, alpha=0.7)
        # plt.tight_layout(pad=0.1)

        # plt.savefig(f"{self.args.attack}_{self.args.dataset}_Split.pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)
        # plt.show()


        return idx_poison_found, idx_clean_found, delta, target_label
    

    def early_train(self, labels, idx_train, verbose, split_epoch, num_attach):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        poison_found=[]
        clean_found = []
        deltas = []
        target_labels = []

        for i in range(split_epoch):
            # print(f"epoch{i}")
            self.train()
    
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = self.rce(output[idx_train], labels[idx_train])

            idx_poison_found, idx_clean_found, delta, target_label = self.split_node(idx_train, labels, i, num_attach)
            
            poison_found.append(idx_poison_found)
            clean_found.append(idx_clean_found)
            deltas.append(delta)
            target_labels.append(target_label)

            loss_train = torch.mean(loss_train)
            loss_train.backward()
            optimizer.step()

        self.eval()
        self.output = output
        return poison_found, clean_found, deltas, target_labels
    

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, attach=None, clean=None, split_epoch=10, target_label=0, num_attach=40):
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

        poison_found, clean_found, deltas, target_labels = self.early_train(self.labels, idx_train, verbose, split_epoch, num_attach)
        return  poison_found, clean_found, deltas, target_labels
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


    

    