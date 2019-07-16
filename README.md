# NeuralNetwork

A super simple neural network that can recognize hand written digits 
(from the MNIST handwritten digit database).

Current Accuracy: _**97.82%**_ (given specific set of parameters)

## Summary

The network is built from scratch (except I used numpy for vector/matrix
operations). The _sigmoid_ function is used as the activation function, and 
the sum of squares error is used as the cost function.

It probably isn't the most efficient way of building the network, this is
just my attempt at learning more about deep learning, etc.


## Usage

The network is built to be trained with any data set, number of layers, etc. However,
`mnist.py` and `main.py` are both built to train and test with the images of handwritten 
digits provided by the mnist database. 

`main.py` has two functions, `recognize` and `draw`. `recognize` will train the network 
with the images, and then test a different set and print out the accuracy rate of how 
well it can classify the digits. `draw` will do the opposite, train the network with the images 
and then draw out what it thinks digits look like (`recognize` runs by default).

If you want to try it out, download the whole repo, running `python3 main.py` 
will create the network, train it, and test it.
