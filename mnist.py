from PIL import Image
import numpy as np

def btoi(b):
	return int.from_bytes(b, "big")

def create_image(pixels, name):
	data = np.array(pixels, dtype=np.uint8)
	image = Image.fromarray(data, "RGB")
	image.save(name)

def save_images(count):
	with open("images-ubyte", "rb") as file:
		metadata = file.read(8)
		width = btoi(file.read(4))
		height = btoi(file.read(4))
		for i in range(count):
			pixels = np.zeros((height, width, 3))
			for w in range(width):
				for h in range(height):
					pixel = file.read(1)
					grayscale = btoi(pixel)
					pixels[w][h] = (grayscale, grayscale, grayscale)
			create_image(pixels, f'tests/test{i}.png')

def read_images(file_path, count):
	images = []
	with open(file_path, "rb") as file:
		metadata = file.read(8)
		width = btoi(file.read(4))
		height = btoi(file.read(4))
		for i in range(count):
			image = [byte / 255 for byte in file.read(width * height)]
			images.append(image)
	return images

def read_labels(file_path, count):
	labels = []
	with open(file_path, "rb") as file:
		metadata = file.read(8)
		for i in range(count):
			label = btoi(file.read(1))
			labels.append(label)
	return labels

def read_labels_gen():
	for i in range(1000000):
		yield i

if __name__ == "__main__":
	for i in range(1000000):
		print(i)
	for i in read_labels_gen():
		print(i)
