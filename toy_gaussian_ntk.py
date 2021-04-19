import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import scipy as sp
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F
import gzip
import pickle
import argparse
import os
import random
import warnings
import sys
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
import datetime, time

torch.backends.cudnn.benchmark=True

class model_ntk(nn.Module):
    def __init__(self, width=100):
        super(model_ntk, self).__init__()
        self.fc1 = nn.Linear(2, width, bias = False) # no bias term
        self.fc2 = nn.Linear(width, 2, bias = False)
        # set the top layer weight to be -1 or +1.
        self.fc2.weight.data = torch.tensor(np.sign(np.random.normal(size = (2, width))).astype("float32"))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
        
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    
def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None]
    
    
def pgd_l2(model, X, y, epsilon, lr, num_iter):
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    delta = torch.zeros_like(X, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()
    for t in range(num_iter):
        loss = -criterion(-model(X + delta), y)
        loss.backward()
        delta.data += lr*delta.grad.detach() #/ norms(delta.grad.detach())
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
        print(loss.item())
        
    return delta.detach().cpu().numpy()


def pgd_l12(model, X, y, B, lr, num_iter):
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    delta = torch.zeros_like(X, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()
    n = X.shape[0]
    for t in range(num_iter):
        loss = criterion(model(X + delta), y)
        loss.backward()
        delta.data += lr*delta.grad.detach() #/ norms(delta.grad.detach())
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        
        if torch.sum(torch.norm(delta.data, dim=1)) > B:
            epsnorm = torch.norm(delta.data, dim=1).detach()
            epsnorm = torch.sort(epsnorm)[0]
            eqn = sum(epsnorm)
            count = 0
            for lam in epsnorm:
                if eqn - (n-count)*lam < B:
                    break
                else:
                    eqn -= lam
                    count += 1
            l = (eqn - B) / (n - count)
            epsnorm = torch.norm(delta.data, dim=1)
            delta.data[epsnorm <= l] = torch.zeros(2).to(device)
            delta.data[epsnorm > l] = (1 - l / epsnorm[epsnorm > l]).unsqueeze(1) * delta.data[epsnorm > l]


        delta.grad.zero_()
        print(t,loss.item(),torch.sum(torch.norm(delta.data, dim=1)))
        
    return delta.detach().cpu().numpy()
    
    
def train(X_train, Y_train, X_val, Y_val, X_test, Y_test, num_epochs, lr, batchsize, width, checkpoint='checkpoint/V.pth'):
    #initialize net
    net = model_ntk(width).to(device)
    print(net)
    # fixed top layer width
    for param in net.fc2.parameters():
        param.requires_grad = False

    best_loss = 1e10
    best_acc = 0
    
    # optimizer and loss
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    #pre-allocate for display
    loss_train = []
    loss_val = []
    loss_test = []

    acc_train = []
    acc_val = []
    acc_test = []

    X_train = torch.from_numpy(X_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    Y_train = torch.from_numpy(Y_train).float().to(device)
    Y_val = torch.from_numpy(Y_val).float().to(device)
    Y_test = torch.from_numpy(Y_test).float().to(device)

    n = len(X_train)
    
    for epoch in range(num_epochs):
        for i in range((n - 1) // batchsize + 1):
            begin = i * batchsize
            end = min(n, begin + batchsize)
            idx = np.arange(begin,end)
            # print(idx)
            
            net.train()  #train mode
            optimizer.zero_grad()

            Y_train_pred = net.forward(X_train[idx])
            loss = criterion(Y_train_pred, Y_train[idx])
            loss.backward()
            optimizer.step()

            loss_train.append(loss.detach())

            net.eval()  #val mode
            with torch.no_grad():
                Y_val_pred = net.forward(X_val)
                Y_test_pred = net.forward(X_test)

                acc_train.append(np.sum(np.argmax(Y_train_pred.detach().cpu().numpy(), axis=1)==np.argmax(Y_train.detach().cpu().numpy(), axis=1))/len(Y_train))
                acc_val.append(np.sum(np.argmax(Y_val_pred.cpu().numpy(), axis=1)==np.argmax(Y_val.detach().cpu().numpy(), axis=1))/len(Y_val))
                acc_test.append(np.sum(np.argmax(Y_test_pred.detach().cpu().numpy(), axis=1)==np.argmax(Y_test.detach().cpu().numpy(), axis=1))/len(Y_test))

                loss_val.append(criterion(Y_val_pred, Y_val))
                loss_test.append(criterion(Y_test_pred, Y_test))

                if best_loss >= loss_val[-1].detach().cpu().numpy():
                    best_loss = loss_val[-1].detach().cpu().numpy()
                    torch.save(net.state_dict(), checkpoint)
            
            print("epoch %d, i %d || Loss train: %1.5f, Loss val: %1.5f, Loss test: %1.5f, acc_train: %1.5f, acc_val: %1.5f, acc_test: %1.5f" % (epoch, i, loss_train[-1], loss_val[-1], loss_test[-1], acc_train[-1], acc_val[-1], acc_test[-1]))

    net.eval() #eval mode - no gradients or weight updates
    net.load_state_dict(torch.load(checkpoint))
    Y_val_pred = net.forward(X_val)
    final_val_test = np.sum(np.argmax(Y_val_pred.detach().cpu().numpy(), axis=1)==np.argmax(Y_val.detach().cpu().numpy(), axis=1))/len(Y_val)
    Y_test_pred = net.forward(X_test)
    final_acc_test = np.sum(np.argmax(Y_test_pred.detach().cpu().numpy(), axis=1)==np.argmax(Y_test.detach().cpu().numpy(), axis=1))/len(Y_test)


    return net, Y_train_pred, loss_train, loss_val, loss_test, acc_train, acc_val, acc_test, final_val_test, final_acc_test
    
    
    

np.random.seed(15)
n_train=1000
x_loc = 5.0
train_n_per_class = int(n_train // 2)
# Create two clusters of data, centered at (+/- x_loc, 0)
clust1 = np.random.normal(loc = (-1*x_loc, 0), size = (train_n_per_class, 2))
clust2 = np.random.normal(loc = (x_loc, 0), size = (train_n_per_class, 2))

n_test = 200
test_n_per_class = int(n_test // 2)
# Create two clusters of data, centered at (+/- x_loc, 0)
clust3 = np.random.normal(loc = (-1*x_loc, 0), size = (test_n_per_class, 2))
clust4 = np.random.normal(loc = (x_loc, 0), size = (test_n_per_class, 2))

# normalize the data
clust1 /= np.linalg.norm(clust1, axis=1)[:,None]
clust2 /= np.linalg.norm(clust2, axis=1)[:,None]
clust3 /= np.linalg.norm(clust3, axis=1)[:,None]
clust4 /= np.linalg.norm(clust4, axis=1)[:,None]

# generate training data and test data
x_train = np.vstack((clust1, clust2)).astype("float32")
y_train = np.concatenate([np.zeros(train_n_per_class),np.ones(train_n_per_class)]).astype("float32")
x_test = np.vstack((clust3, clust4)).astype("float32")
y_test = np.concatenate([np.zeros(test_n_per_class),np.ones(test_n_per_class)]).astype("float32")



date_time = datetime.datetime.utcnow().isoformat().replace(":", "")
    
parser = argparse.ArgumentParser(description='toy Gaussian NTK')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--T', default=1, type=int, help='epoch')
parser.add_argument('--N', default=100, type=int, help='number of runs')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--width', default=200, type=int, help='width of network')
parser.add_argument('--beta', default=0.3, type=float, help='noise rate')
parser.add_argument('--C', default=1000, type=int, help='regime C')
parser.add_argument('--B', default=1, type=float, help='regime B')
parser.add_argument('--regime', default=3, type=int, help='1/2/3')

args = parser.parse_args()

skf = StratifiedKFold(n_splits=5,random_state=40)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
width = args.width
T = args.T
bs = args.bs
lr = args.lr
N = args.N
beta = args.beta
C = args.C
B = args.B



if args.regime==1:
    print("C:", C)
    log_filename = 'log/toy_gaussian_C'+str(args.C)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
    log = open(log_filename, 'w')
    sys.stdout = log
    
if args.regime==2:
    print("B:", B)
    log_filename = 'log/toy_gaussian_B'+str(args.B)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
    log = open(log_filename, 'w')
    sys.stdout = log
    
if args.regime==3:
    print("beta:", beta)
    log_filename = 'log/toy_gaussian_beta'+str(beta)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
    log = open(log_filename, 'w')
    sys.stdout = log
    
    
print("regime:", args.regime)
print("width:", width)
print("epoch T:", T)
print("batchsize:", bs)
print("learning rate:", args.lr)
print("number of trials N:", N)
    
    
# generate poisoned data
#if args.regime !=3:
#    Y_train_onehot = one_hot(np.int16(y_train),2)
#    X_train, Y_train = x_train, Y_train_onehot
#    X_val, Y_val = x_train, Y_train_onehot
#    X_test = x_test
#    Y_test = one_hot(np.int16(y_test),2)
#    net, Y_train_pred, loss_train, loss_val, loss_test, acc_train, acc_val, acc_test, final_val_test, final_acc_test \
#        = train(X_train, Y_train, X_val, Y_val, X_test, Y_test, num_epochs=5000, lr=0.1, batchsize=1000, width=100, checkpoint = 'checkpoint/'+str(date_time)+'.pth')
#
#if args.regime==1:
#    delta = pgd_l12(model=net, X=x_train, y=Y_train_onehot, B=args.C, lr=0.1, num_iter=5000)
#
#if args.regime==2:
#    delta = pgd_l2(model=net, X=x_train, y=Y_train_onehot, epsilon=args.B, lr=1, num_iter=10000)
    



X_test = x_test
Y_test = one_hot(np.int16(y_test),2)
valacc = []
testacc = []
for n in range(N):
    if args.regime==3:
        rvs = sp.stats.bernoulli.rvs(beta, size = n_train)
        idxflip = np.where(rvs==1)[0]
        print(idxflip,len(idxflip))
        y = np.copy(y_train)
        y[idxflip] = (y[idxflip]+1)%2
        advsave = x_train
    else:
        advsave = np.copy(x_train)
        advsave[y_train==0,0] += B
        advsave[y_train==1,0] -= B
#        advsave = x_train + delta
        y = y_train
    idx = np.arange(1000)
    np.random.shuffle(idx)
    advsave = advsave[idx]
    y = y[idx]
    Y_train_onehot = one_hot(np.int16(y),2)
    for idx_train, idx_val in skf.split(advsave,y):
        X_train, Y_train = advsave[idx_train], Y_train_onehot[idx_train]
        X_val, Y_val = advsave[idx_val], Y_train_onehot[idx_val]
        net, Y_train_pred, loss_train, loss_val, loss_test, acc_train, acc_val, acc_test, final_val_test, final_acc_test \
            = train(X_train, Y_train, X_val, Y_val, X_test, Y_test, T, lr, bs, width, 'checkpoint/'+str(date_time)+'.pth')


        valacc.append(final_val_test)
        testacc.append(final_acc_test)
        print("n:{},len(testacc):{}, mean_valacc:{}, mean_testacc:{}".format(n,len(testacc),np.mean(valacc),np.mean(testacc)))
        print("valacc:", valacc)
        print("testacc:", testacc)


