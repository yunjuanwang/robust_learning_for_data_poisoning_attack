'''
Use poisoned data to train the neural network with different width. Contain regime A, B, C. We provide CIFAR10 poisoned data for regime A and regime B.
'''

import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import scipy as sp
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import math
import torch.nn.init as init
import torch.nn.functional as F
from urllib import request
import gzip
import pickle
import argparse
import os
import shutil
import random
import warnings
warnings.filterwarnings('ignore')

from art.config import ART_DATA_PATH
from art.utils import load_dataset, get_file,load_mnist,check_and_transform_label_format
from art.classifiers import KerasClassifier
from art.classifiers import PyTorchClassifier


torch.backends.cudnn.benchmark=True
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
import datetime, time


class model_ntk(nn.Module):
    def __init__(self, width):
        super(model_ntk, self).__init__()
        self.conv_1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv_2 = nn.Conv2d(6,16,5)
        self.fc_1 = nn.Linear(16*5*5,120)
        self.fc_2 = nn.Linear(120, width)
        self.fc_3 = nn.Linear(width,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x
        
        
parser = argparse.ArgumentParser(description='NTK')
parser.add_argument('--T', default=1000, type=int, help='epoch')
parser.add_argument('--N', default=5, type=int, help='number of runs')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--width', default=200, type=int, help='width of network')
parser.add_argument('--beta', default=0.3, type=float, help='corruption rate')
args = parser.parse_args()


# load dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
(x_train, y_train_onehot), (x_test, y_test_onehot), min_, max_ = load_dataset(str("cifar10"))
x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
x_test = x_test.transpose(0, 3, 1, 2).astype("float32")
y_train_onehot = np.float32(y_train_onehot)
y_test_onehot = np.float32(y_test_onehot)
y_train = np.argmax(y_train_onehot, axis=1)
y_test = np.argmax(y_test_onehot, axis=1)


#f=open('regimeA_C300cifar.pkl','rb') # load regime A poisoned data, C=300
f=open('regimeB_B3cifar.pkl','rb') # load regime B poisoned data, B=3
advsave=pickle.load(f)
f.close()



cifar_mu = np.ones((3, 32, 32))
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465
cifar_mu = np.float32(cifar_mu)

cifar_std = np.ones((3, 32, 32))
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616
cifar_std = np.float32(cifar_std)


width = args.width
bs = args.bs
T = args.T
N = args.N
criterion = nn.CrossEntropyLoss()

# change the name of save file
log_filename = 'logfile_'+'cnn_B3_N'+str(N)+'_lr'+str(args.lr)+'_m'+str(width)+'.txt'
log = open(log_filename, 'w')
sys.stdout = log

print("beta:", beta)
print("width:", width)
print("learning rate:", args.lr)
print("batchsize:", bs)
print("epoch T:", T)
print("number of trials N:", N)

date_time = datetime.datetime.utcnow().isoformat().replace(":", "")

skf = StratifiedKFold(n_splits=5,random_state=40)

acc = []
valacc = []
for n in range(N):
    print('n:',n)
    
    # regime C
#    rvs = sp.stats.bernoulli.rvs(beta, size=50000)
#    idxflip = np.where(rvs==1)[0]
#    print(idxflip,len(idxflip))
#    y = np.copy(y_train)
#    y[idxflip] = (y[idxflip]+1)%10
    
    for idx_train, idx_val in skf.split(x_train,y_train):
        
        # regime A or B
        X_train, Y_train = advsave[idx_train], y_train[idx_train]
        X_val, Y_val = advsave[idx_val], y_train[idx_val]
        
        
        # regime C
#        X_train, Y_train = x_train[idx_train], y[idx_train]
#        X_val, Y_val = x_train[idx_val], y[idx_val]
        
        model = model_ntk(width=width).to(device)
        optimizer = optim.SGD(model.parameters(), lr = args.lr)
        classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion,
                            preprocessing=(cifar_mu, cifar_std),
                            optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10,)


        bestacc = 0
        
        for t in range(T):
            classifier.fit(X_train, Y_train, batch_size=bs, nb_epochs=1)
            predictions = classifier.predict(X_val)
            val_acc = np.sum(np.argmax(predictions, axis=1) == Y_val) / len(Y_val)
            train_pred = classifier.predict(X_train)
            train_acc = np.sum(np.argmax(train_pred, axis=1) == Y_train) / len(Y_train)
            
            if val_acc >= bestacc:
                count = 0
                bestacc = max(bestacc, val_acc)
                torch.save(model.state_dict(),'checkpoint/'+str(date_time)+'.pth')
            else:
                count += 1
            if count >=100:
                break
                
            loss = criterion(torch.tensor(train_pred),torch.tensor(Y_train))
            print(n, t, args.lr, val_acc, bestacc, train_acc, loss.item())
            
        valacc.append(bestacc)
        model.load_state_dict(torch.load('checkpoint/'+str(date_time)+'.pth'))
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print(accuracy, valacc, np.mean(valacc))
    
        acc.append(accuracy)
        print(acc,np.mean(acc))
    print(np.mean(valacc), np.mean(acc))
    
