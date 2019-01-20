# Introduction
This Project implements the Multi Layered Perceptron (MLP) from scratch and also demonstrates it's performance on two very famous trivial Datasets (PIMA Indian and MNIST). The main objective of this project is to let people understand the very basics of MLP forward-pass and backward-pass and all the gradient computation, which is the very basic of Deep Learning. 

# Dataset
The two very famous and standard dataset can be found here,
Pima Indian Dataset Source: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
MNIST Dataset Source: http://yann.lecun.com/exdb/mnist

# Approach
Here I have written a seperate python code where I have defined the MLP architecture which has been subsequently used in the two dataset train & test.

# Summary
Pima Indian Dataset:
    Training Accuracy: 71.49 %
    Testing Accuracy : 69.48 %
    
MNIST Dataset:
    Training Accuracy: 98.75 %
    Testing Accuracy : 87.00 %
   
# Features
1. Multilayered Perceptron Implementation using Python and Numpy only to understand the Mathematics behind it
2. Diverse Code which allows user to choose number of Perceptron units
3. Option to save the trained weights and with a very nominal modification into files also
4. Provides Training time loss and accuracy and also plots Loss vs Epoch Curve

# Future Scope
With minor tweaking algorithms other than Stochastic Gradient Descent (i.e- AdaGrad, Adam, RmsProp) can be implemented and performance can be checked. Also with very minor change, the weight initialization technique can also be changed other than Glorot Uniform initialization.
