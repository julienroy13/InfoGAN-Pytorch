# code inspired from : github.com/znxlwm/pytorch-generative-model-collections

import argparse, os
from GAN import GAN
from infoGAN import infoGAN
import torch
import numpy as np


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str,
                        choices=['GAN', 'infoGAN'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'synth', '3Dchairs'],
                        help='The name of dataset', required=True)
    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    return check_args(parser.parse_args())


def check_args(args):
    # creates directories if necessary
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

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

    # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'infoGAN':
        gan = infoGAN(args, SUPERVISED=False)

    # trains the model
    gan.train()