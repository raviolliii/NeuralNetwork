import numpy as np
from time import time
import mnist
from NeuralNet import *

# recognizes digits
def recognize():
	net = NeuralNet(784, 100, 10)

	# train network with mnist data, using pixel values
	# as inputs and numbers as output
	total = 10000
	batch_size = 10
	epochs = 30
	training_set = mnist.load_data("training", total)
	start_time = time()
	net.train(training_set, batch_size, epochs)
	end_time = time()
	print(f"Training Time: {end_time - start_time}s")

	# test using mnist testing data
	total = 1000
	test_data = mnist.load_data("test", total)
	wrong = 0
	for image, label in test_data:
		net.feed(image)
		if net.output != label.index(1):
			wrong += 1
	loss_rate = wrong / total
	acc_rate = 1 - loss_rate
	print(f"Accuracy: {100 * acc_rate}% ({100 * loss_rate}% loss)")


# draws digits
def draw():
	net = NeuralNet(10, 100, 784)

	# train network with mnist data, using numbers as inputs
	# and pixel data as output
	total = 1000
	batch_size = 10
	epochs = 10
	training_set = mnist.load_data("training", total)
	for i, data in enumerate(training_set):
		swapped = (data[1], data[0])
		training_set[i] = swapped
	net.train(training_set, batch_size, epochs)

	# draw out numbers 0-9
	width, height = 28, 28
	for num in range(10):
		input_list = [int(num == i) for i in range(10)]
		net.feed(input_list)
		pixels = np.zeros((height, width, 3))
		for w in range(width):
			for h in range(height):
				output = net.layers[-1]
				pixel = int(output[(w * 28) + h] * 255)
				pixels[w][h] = (pixel, pixel, pixel)
		mnist.create_image(pixels, f'drawings/draw{num}.png')

recognize()
# draw()
