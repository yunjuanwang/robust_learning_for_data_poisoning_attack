'''
generate poisoning CIFAR10 data
'''

import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import scipy as sp
import logging
from scipy.stats import wasserstein_distance
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import math
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F
from urllib import request
import gzip
import pickle
import argparse
import os
import sys
import shutil
import random
import warnings
warnings.filterwarnings('ignore')

from art.config import ART_DATA_PATH
from art.utils import load_dataset, get_file,load_mnist,check_and_transform_label_format
from art.classifiers import PyTorchClassifier
from resnet import *
from alexnet import *

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]
    
    
    
def adv_perturb(X, y, eps, n_label, batchsize, C, S2, eta_eps, model, criterion):
    cifar_mu = np.ones((3, 32, 32))
    cifar_mu[0, :, :] = 0.4914
    cifar_mu[1, :, :] = 0.4822
    cifar_mu[2, :, :] = 0.4465
    cifar_mu = torch.from_numpy(cifar_mu).float().to(device)

    cifar_std = np.ones((3, 32, 32))
    cifar_std[0, :, :] = 0.2471
    cifar_std[1, :, :] = 0.2435
    cifar_std[2, :, :] = 0.2616
    cifar_std = torch.from_numpy(cifar_std).float().to(device)

    X_orig = torch.from_numpy(X).float().to(device)
    X_tensor = (X_orig - cifar_mu) / cifar_std
    y_tensor = torch.from_numpy(y).long().to(device)

    eps_tensor = torch.from_numpy(eps).float().to(device)
    n, c, d, d = X.shape
    B = np.sqrt(n) * C
    print(B)
    
    for s in range(S2):

        for i in range(int((n - 1) // batchsize) + 1):
            idx = random.sample(range(n), k=batchsize)
            epssample = Variable(eps_tensor[idx], requires_grad=True)
            
            pred = model.forward(X_tensor[idx] + epssample)
            loss_eps = -criterion(-pred, y_tensor[idx])
            loss_eps.backward()

            with torch.no_grad():
                epssample += eta_eps * epssample.grad.detach() #/ norms(epssample.grad.detach())  # add the norm thing may help for the regime B (peturbation projected on the L2 ball)
                epssample.grad.zero_()
            eps_tensor[idx] = epssample.detach()

            adv = X_orig[idx] + eps_tensor[idx]
            adv = torch.clamp(adv, min=0, max=1)
            eps_tensor[idx] = adv - X_orig[idx]

        eps_tensor = eps_tensor.reshape(n, c*d*d).detach()
        
        # regime A, project on L21 norm, here C is the corruption rate
        if torch.sum(torch.norm(eps_tensor, dim=1)) > B:
            epsnorm = torch.norm(eps_tensor, dim=1).detach()
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
            
            epsnorm = torch.norm(eps_tensor, dim=1)
            eps_tensor[epsnorm <= l] = torch.zeros(d*d*c).to(device)
            eps_tensor[epsnorm > l] = (1 - l / epsnorm[epsnorm > l]).unsqueeze(1) * eps_tensor[epsnorm > l]

        # regime B, project on L2ball, here C is the maximum L2 perturbation added on each sample
        # eps = eps_tensor.reshape(n,c*d*d).detach().cpu().numpy()
        # tol = 10e-8
        # eps = eps * np.expand_dims(np.minimum(1.0, C / (np.linalg.norm(eps, axis=1) + tol)), axis=1)
        # eps_tensor = torch.tensor(eps).reshape(X.shape).to(device)

        eps_tensor = eps_tensor.reshape(X.shape).detach()

            
        if s%10==0:
            print('s2, step, loss_eps',s, eta_eps, torch.sum(torch.norm(eps_tensor.reshape(n,c*d*d), dim=1)).item(),loss_eps.item())

    return eps_tensor.detach().cpu().numpy()
    
    
def adv_perturb_torch(X, y, Xtest, ytest, n_label, batchsize, C, S1, S2, eta_w, eta_eps, net, criterion):
    n, c, d, d = X.shape

    eps = np.float32(np.zeros(shape=(n,c,d,d)))
    acc = []
    advsave = None
    
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


    model = net().to(device)

#    model.load_state_dict(torch.load("alexnet.pt"))
    optimizer = optim.SGD(model.parameters(), lr = eta_w)
    classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion,
                              preprocessing=(cifar_mu, cifar_std),
                              optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10,)
    for t in range(S1):
        if (t+1) % 20== 0:
            eta_w /= 2
            optimizer = optim.SGD(model.parameters(), lr = eta_w)
            classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion,
                                      preprocessing=(cifar_mu, cifar_std),
                                      optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10,)

        classifier.fit(x_train, y_train, batch_size=batchsize, nb_epochs=1)

    eps = adv_perturb(X, y, eps, n_label, batchsize, C, S2, eta_eps, model, criterion)

    return X + eps



device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Adversarial Attack')
parser.add_argument('--n_label', default=10, type=int,help='number of classes')
parser.add_argument('--S1', default=100, type=int,help='number of rounds for parameter w')
parser.add_argument('--S2', default=300, type=int,help='number of rounds for perturbation eps')
parser.add_argument('--C', default=300, type=float,help='budget for perturbation, proportion of the total norm')
parser.add_argument('--eta_w', default=0.1, type=float,help='step for w')
parser.add_argument('--eta_eps', default=0.3, type=float,help='step for eps')
parser.add_argument('--bs', default=128, type=int,help='batch size')

args = parser.parse_args()

log_filename = 'logfile_alexnet'+'C'+str(args.C)+'S'+str(args.S2)+'eps'+str(args.eta_eps)+'neg.txt'
log = open(log_filename, 'w')
sys.stdout = log


print("n_label:", args.n_label)
print("C:",args.C)
print("S1:",args.S1)
print("S2:",args.S2)
print("eta_w:",args.eta_w)
print("ets_eps:",args.eta_eps)
print("bs:",args.bs)



(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
x_test = x_test.transpose(0, 3, 1, 2).astype("float32")
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


net = AlexNet
criterion = nn.CrossEntropyLoss()
advsave= adv_perturb_torch(x_train, y_train, x_test, y_test, args.n_label, args.bs, args.C, args.S1, args.S2, args.eta_w, args.eta_eps, net, criterion)


f=open('regimeA_C'+str(args.C)+'cifar.pkl','wb')
pickle.dump(advsave,f)
f.close()

model = AlexNet().to(device)
#model.load_state_dict(torch.load("alexnet.pt"))
lr=0.1


optimizer = optim.SGD(model.parameters(), lr = lr)
classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion,
                               preprocessing=(cifar_mu, cifar_std),
                               optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10,)
                               

for t in range(100):
    if (t+1)%20 == 0:
        lr /= 2
        optimizer = optim.SGD(model.parameters(), lr = lr)
        classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion,
                                  preprocessing=(cifar_mu, cifar_std),
                                  optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10,)
        

    
    classifier.fit(advsave,y_train, batch_size=128, nb_epochs=1)
    pred = np.argmax(classifier.predict(x_test),axis=1)
    acc = (pred==y_test).sum()/len(y_test)

    print(t, lr, acc)
    


