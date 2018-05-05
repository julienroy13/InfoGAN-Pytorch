import argparse
import os
from train import train
from gan import GAN
from infogan import infoGAN
import torch
import numpy as np


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

    return check_args(parser.parse_args())


def check_args(args):
    # creates saving directory if necessary
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # makes sure batch_size and epoch are positive
    if args.epoch < 1:
        raise Exception('Number of epochs must be larger than or equal to one')

    if args.batch_size < 1:
        raise Exception('Batch size must be larger than or equal to one')

    return args


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # sets random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # instanciates the model
    if args.model_type == 'GAN':
        model = GAN(args)
    elif args.model_type == 'infoGAN':
        model = infoGAN(args)

    # trains the model
    train(model)