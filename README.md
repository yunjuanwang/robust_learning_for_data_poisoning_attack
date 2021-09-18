# robust_learning_for_data_poisoning_attack

[ICML2021] A PyTorch implementation for the experiments in the paper: [Robust Learning for Data Poisoning Attacks](http://proceedings.mlr.press/v139/wang21r.html)

## Abstract

We investigate the robustness of stochastic approximation approaches against data poisoning attacks. We focus on two-layer neural networks with ReLU activation and show that under a specific notion of separability in the RKHS induced by the infinite-width network, training (finite-width) networks with stochastic gradient descent is robust against data poisoning attacks. Interestingly, we find that in addition to a lower bound on the width of the network, which is standard in the literature, we also require a distribution-dependent upper bound on the width for robust generalization. We provide extensive empirical evaluations that support and validate our theoretical results.

## Implementation

We use adversarial-robustness-toolbox in our code.  Please:  pip install adversarial-robustness-toolbox

We run experiment on GPU.

We put the poisoned data that we generate in [here](https://drive.google.com/drive/folders/1_C5tg3QrmnlS2IIaD4rNSirm4wnf4kcH?usp=sharing).

mnistcorr/ contains the MNIST poisoned data we use for our experiments.

cifarcorr/ contains the CIFAR10 poisoned data we use for our experiments.

1. poison_attack.py: generate poisoned training data.

2. mnist_ntk.py is to use poisoned MNIST data to train neural network with changing width of the top layer. You can either load the poisoned data that we present in mnistcorr folder, or generate corresponding poisoned data to run this experiment for regime A and regime B. 

3. cifar_ntk.py is to use poisoned CIFAR10 data to train neural network with changing width of the top layer. You can either load the poisoned data that we present in cifarcorr folder, or generate corresponding poisoned data to run this experiment for regime A and regime B. 

4. alexnet.py: AlexNet model. 

5. resnet.py: resnet model.


##
If you find it helpful, please cite our paper

```
@inproceedings{wang2021robust,
  title={Robust Learning for Data Poisoning Attacks},
  author={Wang, Yunjuan and Mianjy, Poorya and Arora, Raman},
  booktitle={International Conference on Machine Learning},
  pages={10859--10869},
  year={2021},
  organization={PMLR}
}
```
