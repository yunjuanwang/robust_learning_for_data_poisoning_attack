# robust_learning_for_data_poisoning_attack

We use adversarial-robustness-toolbox in our code. 

Please:  pip install adversarial-robustness-toolbox

We run experiment on GPU.

1. advgenmnist.py is trying to generate poisoned mnist data.  For regime A, we generate poisoned data with C=800, for regime B, we generate poisoned data with B = 3.

2. advgencifar.py is trying to generate poisoned CIFAR10 data. For regime A, we generate poisoned data with C=300, for regime B, we generate poisoned data with B = 3.

3. cv_mnist_ntk.py is to use poisoned MNIST data to train neural network with changing width of the top layer. You need to first generate corresponding poisoned data to run this experiment for regime A and regime B. Please see the comments in the code.

4. cv_cifar_ntk.py is to use poisoned CIFAR10 data to train neural network with changing width of the top layer. You need to first generate corresponding poisoned data to run this experiment for regime A and regime B. Please see the comments in the code.
