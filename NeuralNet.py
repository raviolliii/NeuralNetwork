import math
import numpy as np
import mnist

np.set_printoptions(precision=5, suppress=True)
matrix = np.array

def rand_matrix(n, scale=1):
	rand = np.random.rand(n)
	for i, v in enumerate(rand):
		rand[i] *= scale
	return rand

def sigmoid(x):
	sig = (np.exp(-x)) + 1
	return 1 / sig

def sigmoid_derivative(x):
	sig = sigmoid(x)
	return sig * (1 - sig)

class NeuralNet:
	def __init__(self, *args):
		self.nlayers = len(args)
		self.learning_rate = 3
		# initialize layers and activations
		layers = []
		for n in args:
			rand_array = rand_matrix(n)
			layers.append(rand_array)
		self.layers = matrix(layers)
		# initialize weights and biases
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

	@property
	def output(self):
		output_layer = list(self.layers[-1])
		_output = output_layer.index(max(output_layer))
		return _output

	def train(self, data_set, batch_size, epochs):
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
		delta_weights = [np.zeros(w.shape) for w in self.weights]
		delta_biases = [np.zeros(b.shape) for b in self.biases]
		# feed x (input values) and y (expected values) from data_set
		for x, y in data_set:
			self.feed(x)
			# backpropogate and sum all deltas
			d_weights, d_biases = self.backpropogate(y)
			delta_weights = [total + dw for total, dw in zip(delta_weights, d_weights)]
			delta_biases = [total + db for total, db in zip(delta_biases, d_biases)]
		# adjust all weights/biases with average of deltas
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
		Calculate activations in all hidden layers and output layer,
		using the previous layer's activations, weights, and biases
		"""
		self.layers[0] = matrix(input_values)
		z_values = [np.zeros(l.shape) for l in self.layers]
		for i, layer in enumerate(self.layers[1:], 1):
			weights = self.weights[i - 1]
			biases = self.biases[i - 1]
			a = self.layers[i - 1]
			for j, neuron in enumerate(layer):
				W = weights[j]
				b = biases[j]
				z = (W @ a) + b
				activation = sigmoid(z)
				z_values[i][j] = z
				self.layers[i][j] = activation
		self.z_values = matrix(z_values)

	def backpropogate(self, y_values):
		# calculate error for output layer
		d_weights = [np.zeros(w.shape) for w in self.weights]
		d_biases = [np.zeros(b.shape) for b in self.biases]
		z_values = matrix(self.z_values[-1])
		sig_vector = sigmoid_derivative(z_values)
		delta = (self.layers[-1] - y_values) * sig_vector
		d_weights[-1] = np.outer(delta, self.layers[-2])
		d_biases[-1] = delta
		# propogate through hidden layers
		for l in range(self.nlayers - 2, 0, -1):
			weights = self.weights[l] # W: l - 1
			weighted_delta = weights.T @ delta
			z_values = matrix(self.z_values[l])
			sig_vector = sigmoid_derivative(z_values)
			delta = weighted_delta * sig_vector
			d_weights[l - 1] = np.outer(delta, self.layers[l - 1])
			d_biases[l - 1] = delta
		return (d_weights, d_biases)

	def cost_derivative(self, activations, targets):
		""" partial_cost with respect to partial_activation """
		return activations - targets

	def __repr__(self):
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

