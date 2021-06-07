import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import scipy as sp
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from urllib import request
import pickle
import argparse
import os
import random
import warnings
import sys
warnings.filterwarnings('ignore')

from art.utils import load_dataset, get_file,load_mnist,check_and_transform_label_format
from art.classifiers import KerasClassifier, PyTorchClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
import datetime, time
from poison_attack import adv_perturb, poison_attack

torch.backends.cudnn.benchmark=True

        
# model architecture
class model_ntk(nn.Module):
    def __init__(self, width):
        super(model_ntk, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, width)
        self.fc2 = nn.Linear(width, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

date_time = datetime.datetime.utcnow().isoformat().replace(":", "")
    
parser = argparse.ArgumentParser(description='NTK')
parser.add_argument('--dataset', default="mnist", type=str, help='mnist / cifar')
parser.add_argument('--eta_w', default=0.01, type=float, help='eta_w')
parser.add_argument('--eta_eps', default=0.1, type=float, help='eta_eps')
parser.add_argument('--S1', default=200, type=int, help='S1')
parser.add_argument('--S2', default=500, type=int, help='S2')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--T', default=1000, type=int, help='epoch')
parser.add_argument('--N', default=20, type=int, help='number of runs')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--C', default=500, type=int, help='regime A')
parser.add_argument('--B', default=1, type=float, help='regime B')
parser.add_argument('--beta', default=0.3, type=float, help='noise rate')
parser.add_argument('--width', default=200, type=int, help='width of network')
parser.add_argument('--regime', default=3, type=int, help='1/2/3')
parser.add_argument('--flag', default=0, type=int, help='flag=1: load saved advsave, flag=0, generate poisoned training data')

args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
(x_train, y_train_onehot), (x_test, y_test_onehot), min_, max_ = load_dataset(str("mnist"))
x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
x_test = x_test.transpose(0, 3, 1, 2).astype("float32")
y_train_onehot = np.float32(y_train_onehot)
y_test_onehot = np.float32(y_test_onehot)
y_train = np.argmax(y_train_onehot, axis=1)
y_test = np.argmax(y_test_onehot, axis=1)


width = args.width
bs = args.bs
T = args.T
N = args.N
beta = args.beta
C = args.C
B = args.B
eta_w = args.eta_w
eta_eps = args.eta_eps
S1 = args.S1
S2 = args.S2
flag = args.flag
lr = args.lr
regime = args.regime



if args.regime==1:
    if flag==0:
        #log_filename = 'log/mnist_C'+str(args.C)+'_S1_'+str(S1)+'_etaw'+str(eta_w)+'_S2_'+str(S2)+'_etaeps_'+str(eta_eps)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
        log_filename = 'log/mnist_C'+str(args.C)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
        log = open(log_filename, 'w')
        sys.stdout = log
        
    if flag==1:
        log_filename = 'log/mnist_C'+str(args.C)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
        log = open(log_filename, 'w')
        sys.stdout = log
        f=open('mnistcorr/ntkconv100C'+str(args.C)+'S1000lr0.1.pkl','rb')
        advsave=pickle.load(f)
        f.close()
    print("C:", C)
    
    
if args.regime==2:
    if flag==0:
#        log_filename = 'log/mnist_B'+str(args.B)+'_S1_'+str(S1)+'_etaw'+str(eta_w)+'_S2_'+str(S2)+'_etaeps_'+str(eta_eps)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
        log_filename = 'log/mnist_B'+str(args.B)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
        log = open(log_filename, 'w')
        sys.stdout = log
        
    if flag==1:
        log_filename = 'log/mnist_B'+str(args.B)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
        log = open(log_filename, 'w')
        sys.stdout = log
        f=open('mnistcorr/ntkconv100B'+str(args.C)+'S500lr0.5.pkl','rb')
        advsave=pickle.load(f)
        f.close()
    print("B:", B)
    
if args.regime==3:
    log_filename = 'log/mnist_beta'+str(beta)+'_N'+str(N)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(args.lr)+'_width'+str(width)+'.txt'
    log = open(log_filename, 'w')
    sys.stdout = log
    print("beta:", beta)



print("regime:", args.regime)
print("S1:", args.S1)
print("eta_w:", args.eta_w)
print("S2:", args.S2)
print("eta_eps:", args.eta_eps)


print("width:", width)
print("learning rate:", lr)
print("batchsize:", bs)
print("epoch T:", T)
print("number of trials N:", N)





criterion = nn.CrossEntropyLoss()

X_test = x_test
Y_test = y_test
valacc = []
testacc = []
for n in range(N):
    print('n:',n)
    
    np.random.seed(n)
    random.seed(n)
    np.random.RandomState(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    
    # regime C, generate label flip attack
    if regime==3:
        rvs = sp.stats.bernoulli.rvs(beta, size=60000)
        idxflip = np.where(rvs==1)[0]
        print(idxflip,len(idxflip))
        y = np.copy(y_train)
        y[idxflip] = (y[idxflip]+1)%10
        advsave = x_train
    else:
    # regime A or regime B
        if regime==1:
            advsave = poison_attack(X=x_train, y=y_train, n_label=10, batchsize=bs, C=C, S1=S1, S2=S2, eta_w=eta_w, eta_eps=eta_eps, net=model_ntk(width=100), regime=1, dataset = args.dataset)
        if regime==2:
            advsave = poison_attack(X=x_train, y=y_train, n_label=10, batchsize=bs, C=B, S1=S1, S2=S2, eta_w=eta_w, eta_eps=eta_eps, net=model_ntk(width=100), regime=2, dataset = args.dataset)
            
        y = y_train
        
    
    for idx_train, idx_val in skf.split(advsave,y):
        X_train, Y_train = advsave[idx_train], y[idx_train]
        X_val, Y_val = advsave[idx_val], y[idx_val]
        
        model = model_ntk(width=width).to(device)
        optimizer = optim.SGD(model.parameters(), lr = lr)
        classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion,
                            optimizer=optimizer, input_shape=(1,28,28), nb_classes=10,)


        bestacc = 0
        
        for t in range(T):
            classifier.fit(X_train, Y_train, batch_size=bs, nb_epochs=1)
            val_pred = classifier.predict(X_val)
            val_acc = np.sum(np.argmax(val_pred, axis=1) == Y_val) / len(Y_val)
            
            train_pred = classifier.predict(X_train)
            train_acc = np.sum(np.argmax(train_pred, axis=1) == Y_train) / len(Y_train)
            
            test_pred = classifier.predict(X_test)
            test_acc = np.sum(np.argmax(test_pred, axis=1) == Y_test) / len(Y_test)
            
            if val_acc >= bestacc:
                count = 0
                bestacc = max(bestacc, val_acc)
                torch.save(model.state_dict(),'checkpoint/'+str(date_time)+'.pth')
            else:
                count += 1
            if count >= 30:
                break
                
            train_loss = criterion(torch.tensor(train_pred),torch.tensor(Y_train))
            val_loss = criterion(torch.tensor(val_pred),torch.tensor(Y_val))
            test_loss = criterion(torch.tensor(test_pred),torch.tensor(Y_test))
            print("n:", n, "t:", t, "lr:", lr, "train_loss:", train_loss.item(), "val_loss:", val_loss.item(), "test_loss:", test_loss.item(), "train_acc",train_acc, "valacc:", val_acc, "testacc", test_acc, "best_val_acc", bestacc)
            
        valacc.append(bestacc)
        model.load_state_dict(torch.load('checkpoint/'+str(date_time)+'.pth'))
        predictions = classifier.predict(X_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == Y_test) / len(Y_test)
        testacc.append(accuracy)
        
        print("n:", n, "len(testacc):", len(testacc),"mean_valacc:", np.mean(valacc), "mean_testacc:", np.mean(testacc))
        print("valacc:", valacc)
        print("testacc:", testacc)

