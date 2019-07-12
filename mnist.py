# ===========================================================
# mnist.py
# 
# Helper functions to use the images/labels of handwritten 
# digits provided by the MNIST database.
# ===========================================================

# built in dependencies
from PIL import Image
import numpy as np


def btoi(b):
    """
    Converts a series of bytes into an integer (big endian)
    """
    return int.from_bytes(b, "big")

def create_image(pixels, name):
    """ 
    Creates a .png image using the given pixel data 
    """
    data = np.array(pixels, dtype=np.uint8)
    image = Image.fromarray(data, "RGB")
    image.save(name)

def load_images(file_path, count):
    """ 
    Loads the pixel data for count images from the mnist data,
    into an images array (each with 28x28 grayscale values)
    """
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
    """ 
    Loads the label data for count images from the mnist data,
    into a labels array 
    """
    labels = []
    with open(file_path, "rb") as file:
        metadata = file.read(8)
        for i in range(count):
            label = btoi(file.read(1))
            labels.append(label)
    return labels

def load_data(type, count):
    """ 
    Loads both the image and label data, into a singular array
    where each element is a tuple representing a single piece of 
    data in the structure: (image_pixel_data, expected_output)
    """
    images = load_images(f'data/{type}-images-ubyte', count)
    labels = load_labels(f'data/{type}-labels-ubyte', count)
    data = []
    for image, label in zip(images, labels):
        y_values = [int(i == label) for i in range(10)]
        data.append((image, y_values))
    return data


# testing purposes
if __name__ == "__main__":
    pass
