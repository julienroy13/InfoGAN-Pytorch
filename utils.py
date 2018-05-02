import os
import pdb
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def save_loss_plot(train_history, filename, infogan=False):
    plt.figure(figsize=(10,4))

    # Defines the plot
    plt.plot(train_history['D_loss'], color="blue", label='D loss')
    plt.plot(train_history['G_loss'], color="orange", label='G loss')
    if infogan: 
        plt.plot(train_history['info_loss'], color="pink", label='Info loss')
    plt.title('Training Curves', fontweight='bold')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True, color="lightgrey")

    # Saves the figure
    plt.savefig(filename)
    plt.close()

def generate_samples(gan, z_dim, save_dir, c_cont_dim=None, c_disc_dim=None, seed=1234):
    torch.manual_seed(seed)
    gan.G.eval()

    # Creates samples and saves them
    plt.figure(figsize=(20,20))
    for i in range(25):
        if gan.gpu_mode:
            z = Variable(torch.rand((1, z_dim)).cuda(gan.gpu_id), volatile=True)
        else:
            z = Variable(torch.rand((1, z_dim)), volatile=True)
        
        x = gan.G(z)
        x = x.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()

        plt.subplot(5,5,i+1)
        plt.imshow(x, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(save_dir, 'seed{}.png'.format(seed)), bbox_inches='tight')
    plt.close()