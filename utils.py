import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import pdb

def load_mnist(dataset):
    
    # Loads mnist data from pkl file
    data_dir = os.path.join("data", dataset)
    with open(os.path.join(data_dir, "mnist_data.pkl"), "rb") as f:
        mnist_data = pickle.load(f)

    X = mnist_data['X']
    Y = mnist_data['Y']

    # Reshape to (batch, channels, height, width) and scales in [0., 1.]
    X = X.transpose(0, 3, 1, 2) / 255.

    # Converts X to torch tensor
    X = torch.from_numpy(X).type(torch.FloatTensor)
    Y = torch.from_numpy(Y).type(torch.FloatTensor)
    return X, Y

# WE KEEP THIS FUNCTION (for now) AS A TEMPLATE FOR 3DCHAIRS AND SYNTH DATASETS--------
def load_celebA(dir, transform, batch_size, shuffle):
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader
# -----------------------------------------------------------------------------

def print_network(net):
    # Counts the number of parameters
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    # Prints the architecture
    print('\nNETWORK ARCHITECTURE ------------------------')
    print(net)
    print('Total number of parameters : {}k'.format(int(num_params/1e3)))
    print('-----------------------------------------------\n')

def save_loss_plot(train_history, filename):
    plt.figure(figsize=(10,4))

    # Defines the plot
    plt.plot(train_history['D_loss'], color="blue", label='D loss')
    plt.plot(train_history['G_loss'], color="orange", label='G loss')
    plt.title('Training Curves', fontweight='bold')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True, color="lightgrey")

    # Saves the figure
    plt.savefig(filename)
    plt.close()