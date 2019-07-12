# ===========================================================
# NeuralNet.py
# 
# Full code for a general NeuralNet that uses the 
# sigmoid activation function, and backpropagation as
# a learning algorithm.
# ===========================================================

# built in dependencies
import math
import numpy as np

# numpy configuration/options
np.set_printoptions(precision=5, suppress=True)
matrix = np.array


def sigmoid(x):
    """ 
    Sigmoid function used as the activation function 
    """
    sig = (np.exp(-x)) + 1
    return 1 / sig

def sigmoid_derivative(x):
    """ 
    Derivative of the sigmoid activation function 
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def cost_derivative(activations, targets):
    """ 
    Derivative of the cost function, which is a simple
    sum of squares error
    """
    return activations - targets


class NeuralNet:
    """
    NeuralNet class, handles the training, learning,
    and evaluating of data input
    """

    def __init__(self, *args):
        self.learning_rate = 3
        self.nlayers = len(args)

        # initialize layers and activations
        layers = []
        for n in args:
            rand_array = np.random.uniform(size=(n,))
            layers.append(rand_array)
        self.layers = matrix(layers)

        # initialize weights and biases with values 
        # between the range of [-1/sqrt(d), 1/sqrt(d)],
        # where d is the number of input neurons
        weights, biases = [], []
        for i, n in enumerate(args[1:], 1):
            rows, cols = args[i], args[i - 1]
            low = -(1 / math.sqrt(args[0]))
            high = (1 / math.sqrt(args[0]))
            weights_matrix = np.random.uniform(low, high, (rows, cols))
            biases_matrix = np.random.uniform(low, high, (rows,))
            weights.append(weights_matrix)
            biases.append(biases_matrix)
        self.weights = matrix(weights)
        self.biases = matrix(biases)

    def train(self, data_set, batch_size, epochs):
        # trains the network with the whole data set, splitting
        # into batches and using epochs
        phrase = f'Epoch: [0/{epochs}]'
        total_size = len(data_set)
        for e in range(epochs):
            for i in range(0, total_size, batch_size):
                data_subset = data_set[i:i + batch_size]
                self.train_batch(data_subset, batch_size)
            # print progress
            print("\r" * len(phrase), end="")
            phrase = f'Epoch: [{e + 1}/{epochs}]'
            print(phrase, end="")
        print()

    def train_batch(self, data_set, batch_size):
        """
        Trains the network on a single batch of data.
        Feeds the network, and uses backpropagation to find the 
        total gradient - then averages the gradient values with
        the learning rate before udpating all weights and biases
        in the network
        """
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        # feed x (input values) and y (expected values) from data_set
        for x, y in data_set:
            self.feed(x)
            # backpropogate and add up all deltas
            d_weights, d_biases = self.backpropogate(y)
            delta_weights = [total + dw for total, dw in zip(delta_weights, d_weights)]
            delta_biases = [total + db for total, db in zip(delta_biases, d_biases)]

        # adjust all network weights/biases with average of deltas
        const = self.learning_rate / batch_size
        dw_avg = [const * total for total in delta_weights]
        db_avg = [const * total for total in delta_biases]
        new_weights = [curr - delta for curr, delta in zip(self.weights, dw_avg)]
        new_biases = [curr - delta for curr, delta in zip(self.biases, db_avg)]
        # update all weights/biases
        self.weights = matrix(new_weights)
        self.biases = matrix(new_biases)

    def feed(self, input_values):
        """ 
        Calculate and set activations in all hidden layers and 
        output layer, using the input layer (input values)
        """
        self.layers[0] = matrix(input_values)
        z_values = [np.zeros(l.shape) for l in self.layers]
        for i, layer in enumerate(self.layers[1:], 1):
            weights = self.weights[i - 1]
            biases = self.biases[i - 1]
            a = self.layers[i - 1]
            # each neuron's activation is a weighted sum + its bias
            for j, neuron in enumerate(layer):
                W = weights[j]
                b = biases[j]
                z = (W @ a) + b
                activation = sigmoid(z)
                z_values[i][j] = z
                self.layers[i][j] = activation
        # store the z values (value before sigmoid) for backpropagation
        self.z_values = matrix(z_values)

    def backpropogate(self, y_values):
        """
        Learning process of the network.
        Calculates and returns the gradient values for all the 
        weights and biases, based on the cost function and 
        actual vs desired output values
        """
        d_weights = [np.zeros(w.shape) for w in self.weights]
        d_biases = [np.zeros(b.shape) for b in self.biases]

        # calculate error for output layer
        z_values = matrix(self.z_values[-1])
        sig_vector = sigmoid_derivative(z_values)
        cost_delta = cost_derivative(self.layers[-1], y_values)
        delta = cost_delta * sig_vector
        d_weights[-1] = np.outer(delta, self.layers[-2])
        d_biases[-1] = delta

        # propogate error through hidden layers
        for l in range(self.nlayers - 2, 0, -1):
            weights = self.weights[l] # W: l - 1
            weighted_delta = weights.T @ delta
            z_values = matrix(self.z_values[l])
            sig_vector = sigmoid_derivative(z_values)
            delta = weighted_delta * sig_vector
            d_weights[l - 1] = np.outer(delta, self.layers[l - 1])
            d_biases[l - 1] = delta
        return (d_weights, d_biases)

    @property
    def output(self):
        """
        output value is the index in the output layer
        that has the highest activation
        """
        output_layer = list(self.layers[-1])
        _output = output_layer.index(max(output_layer))
        return _output

    def __repr__(self):
        """ 
        String representation of NeuralNet 
        """
        res = "\n" + ("=" * 40)
        res += "\nNetwork: \n\n"
        res += "Layers: \n"
        for layer in self.layers:
            res += str(layer) + "\n"
        res += "\nWeights: \n"
        for weight in self.weights:
            res += str(weight) + "\n"
        res += "\nBiases: \n"
        for bias in self.biases:
            res += str(bias) + "\n"
        res += ("=" * 40) + "\n"
        return res

