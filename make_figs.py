import argparse
import os
from model import MODEL
from train import train
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import os
import pickle
import torch
from torch.autograd import Variable


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_type', type=str, choices=['GAN', 'infoGAN'], required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'synth', '3Dchairs'], required=True)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # instanciates the model
    model = MODEL(args)

    # trains the model
    train(model)







# ===================================
# The folowing functions are not used during training.
# Only used to generate figures by running make_figs.py directly

def generate_figure1(model, filename):
    model.G.eval()

    # Creates samples and saves them
    plt.figure(figsize=(20,8))

    z_dim = 62
    n_samples = 5
    c_disc_dim = 10
    c_cont_dim = 2
    k=1
    for i in range(n_samples):

        # Basic z vector
        z = torch.rand((1, z_dim))

        # Continuous code
        c_cont = torch.from_numpy(np.random.uniform(-1, 1, size=(1, c_cont_dim))).type(torch.FloatTensor)

        # Converts to Variable (sends to GPU if necessary)
        if model.gpu_mode:
            z = Variable(z.cuda(model.gpu_id), volatile=True)
            c_cont = Variable(c_cont.cuda(model.gpu_id), volatile=True)
        else:
            z = Variable(z, volatile=True)
            c_cont = Variable(c_cont, volatile=True)

        for j in range(n_disc_codes):
            
            # Discrete code
            c_disc = torch.from_numpy(np.zeros(shape=(1,n_disc_codes))).type(torch.FloatTensor)
            c_disc[j] = 1.

            # Converts to Variable (sends to GPU if necessary)
            if model.gpu_mode:
                c_disc = Variable(c_disc.cuda(model.gpu_id), volatile=True)
            else:
                c_disc = Variable(c_disc, volatile=True)

            # Forward propagation
            x = model.G(z, c_cont, c_disc)

            # Reshapes dimensions and convert to ndarray
            x = x.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()

            # Adds sample to the figure
            plt.subplot(n_samples, c_disc_dim, k)
            plt.imshow(x, cmap='gray')
            plt.axis('off')
            k += 1

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def generate_figure2(model, filename):
    pass

def generate_figure3(model, filename):
    pass

if __name__ == "__main__":

    pass