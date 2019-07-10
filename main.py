import mnist
from NeuralNet import *
from pprint import pprint

# create the net with layer sizes, and train
net = NeuralNet(784, 100, 10)
net.train(30, 10000, 10)

# testing
total = 1000
images = mnist.read_images("test-images-ubyte", total)
labels = mnist.read_labels("test-labels-ubyte", total)

wrong = 0
for image, label in zip(images, labels):
	net.feed(image)
	if net.output != label:
		wrong += 1
print(f'Loss Rate: {100 * wrong / total}%')
