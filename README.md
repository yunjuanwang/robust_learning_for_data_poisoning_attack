# robust_learning_for_data_poisoning_attack

We use adversarial-robustness-toolbox in our code.  Please:  pip install adversarial-robustness-toolbox

We run experiment on GPU.

mnistcorr/ contains the MNIST poisoned data we use for our experiments.

cifarcorr/ contains the CIFAR10 poisoned data we use for our experiments.

1. toy_gaussian_ntk.py: A toy example. FIrst generate 2-dimensional binary training data and test data using two Gaussian distribution with center located at (-5,0) and (5,0). Then normalize the data so that |x|=1. We use two-layer ReLU network with fixed top layer. To validate regime B, we add perturbation B = 0.975 to each of the samples, making the training data to move horizontally either left or right. To validate regime C, we random flip labels with probability beta = 0.4.

2. poison_attack.py: generate poisoned training data.

3. mnist_ntk.py is to use poisoned MNIST data to train neural network with changing width of the top layer. You can either to first generate corresponding poisoned data to run this experiment for regime A and regime B, or load the poisoned data that we present in mnistcorr folder. Please see the comments in the code.

4. cifar_ntk.py is to use poisoned CIFAR10 data to train neural network with changing width of the top layer. You can either first generate corresponding poisoned data to run this experiment for regime A and regime B, or load the poisoned data that we present in cifarcorr folder. Please see the comments in the code.

5. alexnet.py: AlexNet model. 


