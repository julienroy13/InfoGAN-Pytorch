import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import os
import pdb

IMAGE_SIZE = 28

# IMPORTANT : this file needs to be run from : InfoGAN-Pytorch/data/
# This file also assumes that you have previously downloaded the .tar.gz raw files from
#    Yann Lecun website and placed them in InfoGAN-Pytorch/data/raw/

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

# Creates a folder if necessary
if not os.path.exists('mnist'):
	os.makedirs('mnist')

# Extract MNIST into np arrays.
train_data = extract_data(os.path.join("raw", "train-images-idx3-ubyte.gz"), 60000)
test_data = extract_data(os.path.join("raw", "t10k-images-idx3-ubyte.gz"), 10000)

train_labels = extract_labels(os.path.join("raw", "train-labels-idx1-ubyte.gz"), 60000)
test_labels = extract_labels(os.path.join("raw", "t10k-labels-idx1-ubyte.gz"), 10000)

mnist_data = {}
mnist_data['X'] = np.concatenate([train_data, test_data], axis=0)
mnist_data['Y'] = np.concatenate([train_labels, test_labels], axis=0)

print("mnist_data['X'].shape : {}".format(mnist_data['X'].shape))
print("mnist_data['Y'].shape : {}".format(mnist_data['Y'].shape))

with open(os.path.join("mnist", "mnist_data.pkl"), "wb") as f:
    pickle.dump(mnist_data, f, protocol=pickle.HIGHEST_PROTOCOL)