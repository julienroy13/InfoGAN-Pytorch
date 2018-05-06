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


# ===================================
# The folowing functions are not used during training.
# Only used to generate figures by running make_figs.py directly

def generate_figure1(model, filename):
    """Shows the effect of varying the discrete codes"""
    model.G.eval()

    # Creates samples and saves them
    plt.figure(figsize=(20,8))

    z_dim = model.z_dim
    n_samples = 5
    c_disc_dim = model.c_disc_dim
    c_cont_dim = model.c_cont_dim
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

        for j in range(c_disc_dim):
            
            # Discrete code
            c_disc = torch.from_numpy(np.zeros(shape=(1,c_disc_dim))).type(torch.FloatTensor)
            c_disc[0, j] = 1.

            # Converts to Variable (sends to GPU if necessary)
            if model.gpu_mode:
                c_disc = Variable(c_disc.cuda(model.gpu_id), volatile=True)
            else:
                c_disc = Variable(c_disc, volatile=True)

            # Forward propagation
            x = model.G(z, c_cont, c_disc, model.dataset)

            # Reshapes dimensions and convert to ndarray
            x = x.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()

            # Adds sample to the figure
            plt.subplot(n_samples, c_disc_dim, k)
            plt.imshow(x, cmap='gray')
            plt.axis('off')
            k += 1

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def generate_figure2(model, filename, c_idx):
    """Shows the effect of varying a continuous code (c_idx)"""
    model.G.eval()

    # Creates samples and saves them
    plt.figure(figsize=(20,8))

    z_dim = model.z_dim
    n_samples = 5
    c_disc_dim = model.c_disc_dim
    c_cont_dim = model.c_cont_dim
    k=1
    for i in range(n_samples):

        # Basic z vector
        z = torch.rand((1, z_dim))

        # Discrete code
        c_disc = torch.from_numpy(np.random.multinomial(1, model.c_disc_dim * [float(1.0 / model.c_disc_dim)], size=[1])).type(torch.FloatTensor)
        for j in range(model.n_disc_code-1):
            c_disc = torch.cat([c_disc, torch.from_numpy(np.random.multinomial(1, model.c_disc_dim * [float(1.0 / model.c_disc_dim)], size=[1])).type(torch.FloatTensor)], dim=1)

        # Converts to Variable (sends to GPU if necessary)
        if model.gpu_mode:
            z = Variable(z.cuda(model.gpu_id), volatile=True)
            c_disc = Variable(c_disc.cuda(model.gpu_id), volatile=True)
        else:
            z = Variable(z, volatile=True)
            c_disc = Variable(c_disc, volatile=True)

        for j in range(10):
            
            # Continuous code
            c_cont = torch.from_numpy(np.array([[0., 0.]])).type(torch.FloatTensor)
            c_cont[0, c_idx] = -1. + (0.2 * j)

            # Converts to Variable (sends to GPU if necessary)
            if model.gpu_mode:
                c_cont = Variable(c_cont.cuda(model.gpu_id), volatile=True)
            else:
                c_cont = Variable(c_cont, volatile=True)

            # Forward propagation
            x = model.G(z, c_cont, c_disc, model.dataset)

            # Reshapes dimensions and convert to ndarray
            x = x.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()

            # Adds sample to the figure
            plt.subplot(n_samples, 10, k)
            plt.imshow(x, cmap='gray')
            plt.axis('off')
            k += 1

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def generate_figure3(model, filename):
    pass



if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # instanciates the model
    model = MODEL(args, test_only=True)

    # trains the model
    model.load()

    # creates the figure directory
    file_dir = os.path.join("figures", model.dataset, model.model_type)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # sets the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # generates the figures
    generate_figure1(model, filename=os.path.join(file_dir, "figure1.png"))
    generate_figure2(model, filename=os.path.join(file_dir, "figure2_c0.png"), c_idx=0)
    generate_figure2(model, filename=os.path.join(file_dir, "figure2_c1.png"), c_idx=1)