from PIL import Image
import numpy as np

def btoi(b):
	return int.from_bytes(b, "big")

def create_image(pixels, name):
	data = np.array(pixels, dtype=np.uint8)
	image = Image.fromarray(data, "RGB")
	image.save(name)

def save_images(file_path, count, output_path_format):
	with open(file_path, "rb") as file:
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
			create_image(pixels, output_path_format.format(i))

def load_images(file_path, count):
	images = []
	with open(file_path, "rb") as file:
		metadata = file.read(8)
		width = btoi(file.read(4))
		height = btoi(file.read(4))
		for i in range(count):
			image = [byte / 255 for byte in file.read(width * height)]
			images.append(image)
	return images

def load_labels(file_path, count):
	labels = []
	with open(file_path, "rb") as file:
		metadata = file.read(8)
		for i in range(count):
			label = btoi(file.read(1))
			labels.append(label)
	return labels

def load_data(type, count):
	images = load_images(f'data/{type}-images-ubyte', count)
	labels = load_labels(f'data/{type}-labels-ubyte', count)
	data = []
	for image, label in zip(images, labels):
		y_values = [int(i == label) for i in range(10)]
		data.append((image, y_values))
	return data

if __name__ == "__main__":
	pass
