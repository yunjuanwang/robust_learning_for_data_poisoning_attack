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
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
import datetime, time

torch.backends.cudnn.benchmark=True


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


def adv_perturb(X, y, eps, n_label, batchsize, C, S2, eta_eps, model, criterion, regime=1):
    """
    X,y: training data.
    n_label: the number of labels.
    batchsize: batch size.
    C: perturbed parameter, for regime A, the overall budget is C*sqrt{n}, for regime B, the per-sample budget is C.
    S2: epoch for generating poisoned data.
    eta_eps: learning rate for generating poisoned data.
    model: model.
    criterion: loss function.
    regime: 1: regime A; 2: regime B.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_orig = torch.from_numpy(X).float().to(device)
    X_tensor = X_orig
    y_tensor = torch.from_numpy(y).long().to(device)

    eps_tensor = torch.from_numpy(eps).float().to(device)
    n, c, d, d = X.shape
    B = np.sqrt(n) * C # L_21 norm of the perturbation budget
    print(B)
    
    for s in range(S2):
        for i in range(int((n - 1) // batchsize) + 1):
            idx = random.sample(range(n), k=batchsize)
            epssample = Variable(eps_tensor[idx], requires_grad=True)
            
            pred = model.forward(X_tensor[idx] + epssample)
            loss_eps = -criterion(-pred, y_tensor[idx]) # use negated loss
            loss_eps.backward()

            with torch.no_grad():
                if regime==1:
                    epssample += eta_eps * epssample.grad.detach() 
                if regime==2:
                    epssample += eta_eps * epssample.grad.detach() / norms(epssample.grad.detach())  # devide the gradient by its norm may help for the regime B (peturbation projected on the L2 ball)
                epssample.grad.zero_()
            eps_tensor[idx] = epssample.detach()

            adv = X_orig[idx] + eps_tensor[idx]
            adv = torch.clamp(adv, min=0, max=1)
            eps_tensor[idx] = adv - X_orig[idx]

        eps_tensor = eps_tensor.reshape(n, c*d*d).detach()
        
        # regime A, project on L21 norm, here C is the corruption rate
        if regime==1:
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
                
        if regime==2:
        # regime B, project on L2ball, here C is the maximum L2 perturbation added on each sample
             eps = eps_tensor.reshape(n,c*d*d).detach().cpu().numpy()
             tol = 10e-8
             eps = eps * np.expand_dims(np.minimum(1.0, C / (np.linalg.norm(eps, axis=1) + tol)), axis=1)
             eps_tensor = torch.tensor(eps).reshape(X.shape).to(device)

        eps_tensor = eps_tensor.reshape(X.shape).detach()

            
        if (s+1)%10==0:
            print('s2, step, loss_eps',s, eta_eps, torch.sum(torch.norm(eps_tensor.reshape(n,c*d*d), dim=1)).item(),loss_eps.item())

    return eps_tensor.detach().cpu().numpy()
    
    
def poison_attack(X, y, n_label, batchsize, C, S1, S2, eta_w, eta_eps, net, regime=1, dataset="mnist"):
    """
    X,y: training data.
    n_label: the number of labels.
    batchsize: batch size.
    C: perturbed parameter, for regime A, the overall budget is C*sqrt{n}, for regime B, the per-sample budget is C.
    S1: epoch for training classifier using clean data, to get the best model weight
    eta_w: learning rate for training classifier using clean data
    S2: epoch for generating poisoned data
    eta_eps: learning rate for generating poisoned data
    net: network
    criterion: loss function
    regime: 1: regime A; 2: regime B
    """
    n, c, d, d = X.shape

    eps = np.float32(np.zeros(shape=(n,c,d,d)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = net.to(device)
    if dataset == "mnist":
        model.load_state_dict(torch.load('ntkconv100_baseline.pth'))
    if dataset == "cifar":
        model.load_state_dict(torch.load('alexnet_baseline.pth'))
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = eta_w)
    
    # MNIST
    #classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion, optimizer=optimizer, input_shape=(1,28,28), nb_classes=10,)
    #CIFAR
#    cifar_mu = np.ones((3, 32, 32))
#    cifar_mu[0, :, :] = 0.4914
#    cifar_mu[1, :, :] = 0.4822
#    cifar_mu[2, :, :] = 0.4465
#
#    cifar_std = np.ones((3, 32, 32))
#    cifar_std[0, :, :] = 0.2471
#    cifar_std[1, :, :] = 0.2435
#    cifar_std[2, :, :] = 0.2616
#    classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion, preprocessing=(cifar_mu, cifar_std), optimizer=optimizer, input_shape=(3,32,32), nb_classes=10)

#    classifier.fit(X, y, batch_size=batchsize, nb_epochs=S1)

    eps = adv_perturb(X, y, eps, n_label, batchsize, C, S2, eta_eps, model, criterion, regime)

    return X + eps
    
        
        
