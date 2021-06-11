# robust_learning_for_data_poisoning_attack

We use adversarial-robustness-toolbox in our code.  Please:  pip install adversarial-robustness-toolbox

We run experiment on GPU.

We put the poisoned data that we generate and use for test accuracy under different width in [here](https://drive.google.com/drive/folders/1_C5tg3QrmnlS2IIaD4rNSirm4wnf4kcH?usp=sharing).

mnistcorr/ contains the MNIST poisoned data we use for our experiments.

cifarcorr/ contains the CIFAR10 poisoned data we use for our experiments.

1. poison_attack.py: generate poisoned training data.

2. mnist_ntk.py is to use poisoned MNIST data to train neural network with changing width of the top layer. You can either to first generate corresponding poisoned data to run this experiment for regime A and regime B, or load the poisoned data that we present in mnistcorr folder. Please see the comments in the code.

3. cifar_ntk.py is to use poisoned CIFAR10 data to train neural network with changing width of the top layer. You can either first generate corresponding poisoned data to run this experiment for regime A and regime B, or load the poisoned data that we present in cifarcorr folder. Please see the comments in the code.

4. alexnet.py: AlexNet model. 

5. resnet.py: resnet model.


