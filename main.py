import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import time
import torch.nn.functional as Fxiwnag
import torch
import math
import numpy as np
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from utils import *
from metrics import *
from generator import  *
import matplotlib
matplotlib.rc("font",family='WenQuanYi Micro Hei')
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from torch.autograd import Variable
path ='/home/lcf/Desktop/EEG open dataset/ASU/'
dataset = '60/8'
# path ='/home/lcf/Desktop/SPL/'
np.random.seed(100)

# labels_name =['你','去','天','头','来','水','说']
n_splits = 8
import sys
import torch

n_features = 500
# hidden_dim = 4

epochs = 20000
early_stop = 10
weight_decay = 5e-4
learning_rate =  0.0001
batch_size = 40

tx, ty,ta = load_test_adj(path,dataset)
print('tx',tx.shape)
x_name, a_name = load_adj(path,dataset)
num_nodes = tx.shape[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FConvolution(nn.Module):
    def __init__(self, in_features, num_nodes, out_features,bias:float = 0.0):
        super(FConvolution,self).__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features * self.num_nodes, self.num_nodes * out_features)
        )
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.num_nodes * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, adj):
        bat_long = inputs.size(0)
        inputs = torch.reshape(inputs, (-1, self.num_nodes, self.in_features))
        support = torch.bmm(adj, inputs)
        support = torch.reshape(support, (bat_long, self.num_nodes * self.in_features))
        output = torch.mm(support, self.weight)
        outputs = torch.reshape(output, (bat_long, self.out_features,self.num_nodes, ))

        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs

class GConvolution(nn.Module):
    def __init__(self,in_features, num_nodes,out_features,bias: float = 0.0):
        super(GConvolution,self).__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor( in_features * self.num_nodes, self.in_features * out_features)
        )
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor( self.num_nodes * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    def forward(self, inputs, adj):

        fg = torch.bmm(inputs,adj)
        temp = torch.reshape(fg,(-1, self.num_nodes * self.in_features))
        te = torch.mm(temp,self.weight)
        outputs = torch.reshape(te,(-1,self.in_features,self.out_features))


        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs

class FGCN(nn.Module):
    def __init__(self, seq_len,num_nodes,f_out,f2_out, out_size):
        super(FGCN, self).__init__()

        self.out_size = out_size
        self.f_out = f_out
        self.seq_long = seq_len
        self._num_nodes = num_nodes
        self.f2_out  = f2_out
        self.fgcn1  = FConvolution(seq_len, num_nodes,f_out)
        self.fgcn2 = FConvolution(f_out, num_nodes,f2_out )

        self.gcn1  = GConvolution(f2_out,num_nodes,out_size)
        # self.gcn2 = GConvolution(seq_len,out_size,  8)

        self.fc = nn.Linear(self.out_size * self.f2_out, 7)
        self.dropout = nn.Dropout(p = 0.15)


    def forward(self, X, adj):
        A1 = F.relu(self.fgcn1 (X, adj))
        A1 = self.dropout(A1)
        A2 = F.relu(self.fgcn2 (A1, adj))

        G1 = F.tanh( self.gcn1(A2, adj))


        G1 = self.dropout(G1)

        s = torch.reshape(G1,(-1,self.out_size * self.f2_out))

        # if X.shape[0]> 70:
        #     np.savetxt( '/home/lcf/Desktop/EEG open dataset/fc/1/fc-1-%d.csv',s, delimiter=',')

        out = self.fc(s)

        # return out
        return F.log_softmax(out, dim = 1)

model  = FGCN(512,60,196,132,16)

model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction='none').to(device)
optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay = weight_decay)

# labels_name =['cooperate','in','out','up'] #1
# labels_name =['cooperate','independent','in','up','out']  #3

kf = KFold(n_splits = n_splits)
torch.cuda.empty_cache()
for i in range(epochs):
    for x, y, adjs in generator_data(path, dataset,batch_size,x_name,a_name):
        train_ids = TensorDataset(x, y, adjs)
        train_loader = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)
        for train_index, data in enumerate(train_loader, 1):
            t_sum = 0
            v_sum = 0
            x_data, x_label, x_adj = data
            x_data = x_data.to(device)
            x_label = x_label.to(device)
            x_adj = x_adj.to(device)
            for train_index,valid_index in kf.split(x_data):
                train_data = x_data[train_index]  #y_test is train label
                train_labels1 = x_label[train_index]

                train_adj1 = x_adj[train_index]

                val_data = x_data[valid_index]
                val_labels1 = x_label[valid_index]
                val_adj1 = x_adj[valid_index]
                # print('train:%s,valid:%s'%(train_index,valid_index))
                train_acc = 0
                train_total = 0

                output = model(train_data.data, train_adj1)
                loss = criterion(output, train_labels1)
                optimizer.zero_grad()
                loss.backward(torch.ones_like(loss))
                optimizer.step()

                q, pre = torch.max(output, dim=1)

                train_total += train_labels1.size(0)
                train_acc += (pre == train_labels1).sum().item()
                s1 = train_acc / train_total
                t_sum = t_sum + s1

                model.eval()
                v_correct = 0
                v_total = 0
                v_labels2 = val_labels1
                val_data = val_data
                v_outputs = model(val_data.data, val_adj1)
                v_loss = criterion(v_outputs, v_labels2)
                v_, v_predicted = torch.max(v_outputs, 1)
                v_total += v_labels2.size(0)
                v_correct += (v_predicted == v_labels2).sum().item()
                s2 = v_correct / v_total
                v_sum = v_sum + s2
                # print('train acc', s1, 'val acc',s2)

            # print(i, 'train loss', t_sum / n_splits, 'val loss', v_sum / n_splits)
            print(i,  'train acc',t_sum/n_splits, 'val acc', v_sum/n_splits )
            with torch.no_grad():
                correct = 0
                total = 0
                labels = ty
                outputs = model(tx.data, ta)  # ,tadjs
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('test acc--------------------------------------------------------', correct / total)

                pred_y = predicted
                accs = (correct / total)

                cm = confusion_matrix(labels, pred_y )

                print(cm)
                # plot_confusion_matrix(cm,target_names =labels_name,m=i,acc=accs)

