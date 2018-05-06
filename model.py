import utils
import torch
import time
import os
import pickle
import pdb
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

class generator(nn.Module):
    def __init__(self, latent_dim, output_height, output_width, output_features, dataset):
        super(generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_height = output_height
        self.output_width = output_width
        self.output_features = output_features

        if dataset == "mnist":
            # First layers are fully connected
            self.fc_part = nn.Sequential(
                nn.Linear(self.latent_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                nn.Linear(1024, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(),
            )

            # Then we switch to deconvolution (transpose convolutions)
            self.deconv_part = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_features, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(), #[0, 1] images
            )

        elif dataset == "3Dchairs":
            # First layers are fully connected
            self.fc_part = nn.Sequential(
                nn.Linear(self.latent_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                nn.Linear(1024, 256 * 6 * 6), nn.BatchNorm1d(256 * 6 * 6), nn.ReLU(),
            )

            # Then we switch to deconvolution (transpose convolutions)
            self.deconv_part = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_features, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(), #[0, 1] images
            )

        else:
            raise NotImplemented
        
        self.initialize_weights()

    def forward(self, z, cont_code, discr_code, dataset):
        # Concatenates latent vector and latent codes (continuous and discrete)
        x = torch.cat([z, cont_code, discr_code], dim=1)
        
        # Forwards through first fully connected layers
        x = self.fc_part(x)
        
        # Reshapes into feature maps 4 times smaller than original size
        if dataset == "mnist":
            x = x.view(-1, 128, 7, 7)
        elif dataset == "3Dchairs":
            x = x.view(-1, 256, 6, 6)
        else:
            raise NotImplemented
        
        # Feedforward through deconvolutional part (upsampling)
        x = self.deconv_part(x)

        # Makes sure the shape is right
        if dataset == "mnist":
            assert x.size(2)==28 and x.size(3)==28, "Input to discriminator for {} dataset not of shape (batch,channels,28,28)".format(dataset)
        elif dataset == "3Dchairs":
            assert x.size(2)==64 and x.size(3)==64, "Input to discriminator for {} dataset not of shape (batch,channels,64,64)".format(dataset)
        else:
            raise NotImplemented

        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()

class discriminator(nn.Module):
    def __init__(self, input_height, input_width, input_features, output_dim, len_discrete_code, len_continuous_code, dataset):
        super(discriminator, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_features = input_features
        self.output_dim = output_dim
        self.len_discrete_code = len_discrete_code
        self.len_continuous_code = len_continuous_code

        if dataset == "mnist":
            # First layers are convolutional (subsampling)
            self.conv_part = nn.Sequential(
                nn.Conv2d(self.input_features, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            )

            # Then we switch to fully connected before sigmoidal output unit
            self.fc_part = nn.Sequential(
                nn.Linear(128 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            )

        elif dataset == "3Dchairs":
            # First layers are convolutional (subsampling)
            self.conv_part = nn.Sequential(
                nn.Conv2d(self.input_features, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            )

            # Then we switch to fully connected before sigmoidal output unit
            self.fc_part = nn.Sequential(
                nn.Linear(256 * 6 * 6, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            )

        else:
            raise NotImplemented
        
        self.initialize_weights()

    def forward(self, x, dataset):
        # Makes sure the shape is right
        if dataset == "mnist":
            assert x.size(2)==28 and x.size(3)==28, "Input to discriminator for {} dataset not of shape (batch,channels,28,28)".format(dataset)
        elif dataset == "3Dchairs":
            assert x.size(2)==64 and x.size(3)==64, "Input to discriminator for {} dataset not of shape (batch,channels,64,64)".format(dataset)
        else:
            raise NotImplemented

        # Feedforwards through convolutional (subsampling) layers
        y = self.conv_part(x)

        # Reshapes as a vector
        if dataset == "mnist":
            y = y.view(-1, 128 * 7 * 7)
        elif dataset == "3Dchairs":
            y = y.view(-1, 256 * 6 * 6)
        else:
            raise NotImplemented

        # Feedforwards through fully connected layers
        y = self.fc_part(y)

        # D output
        a = F.sigmoid(y[:, 0])

        # Q outputs
        b = y[:, 1:1+self.len_continuous_code] # continuous codes
        c = F.softmax(y[:, 1+self.len_continuous_code:])  # discrete codes

        return a, b, c

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()

class MODEL(object):
    def __init__(self, args, test_only=False):
        self.model_type = args.model_type
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.save_dir = os.path.join(args.save_dir, self.dataset, self.model_type)
        self.gpu_mode = args.gpu_mode
        self.gpu_id = args.gpu_id
        self.test_only = test_only

        if self.model_type == "infoGAN":
            is_infogan = True
        else:
            is_infogan = False

        # Defines input/output dimensions
        if self.dataset == 'mnist':
            self.x_height = 28
            self.x_width = 28
            self.x_features = 1
            self.y_dim = 1
            self.n_disc_code = 1
            self.c_disc_dim = 10 
            self.c_cont_dim = 2  
            self.c_dim = self.n_disc_code*self.c_disc_dim + self.c_cont_dim
            self.z_dim = 62

        elif self.dataset == '3Dchairs':
            self.im_resize = 64
            self.x_height = self.im_resize
            self.x_width = self.im_resize
            self.x_features = 1
            self.y_dim = 1
            self.n_disc_code = 3
            self.c_disc_dim = 20
            self.c_cont_dim = 1
            self.c_dim = self.n_disc_code*self.c_disc_dim + self.c_cont_dim
            self.z_dim = 128

        elif self.dataset == 'synth':
            self.im_resize = 128
            self.x_height = self.im_resize
            self.x_width = self.im_resize
            self.x_features = 1
            self.y_dim = 1
            self.n_disc_code = 1
            self.c_disc_dim = 10
            self.c_cont_dim = 3
            self.c_dim = self.n_disc_code*self.c_disc_dim + self.c_cont_dim
            self.z_dim = 62

        else:
            raise Exception('Unsupported dataset')

        # Initializes the models and their optimizers
        self.G = generator(self.z_dim + self.c_dim, self.x_height, self.x_width, self.x_features, self.dataset)
        utils.print_network(self.G)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        
        if not test_only:
            self.D = discriminator(self.x_height, self.x_width, self.x_features, self.y_dim, self.n_disc_code*self.c_disc_dim, self.c_cont_dim, self.dataset)
            utils.print_network(self.D)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        
            self.info_optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

            # Loss functions
            self.BCE_loss = nn.BCELoss()

            if is_infogan:
                self.MSE_loss = nn.MSELoss()
                self.CE_losses = []
                for i in range(self.n_disc_code):
                    self.CE_losses.append(nn.CrossEntropyLoss())

        # Sends the models of GPU (if defined)
        if self.gpu_mode:
            self.G.cuda(self.gpu_id)
            if not test_only: 
                self.D.cuda(self.gpu_id)
            
                if is_infogan:
                    self.BCE_loss.cuda(self.gpu_id)
                    self.MSE_loss.cuda(self.gpu_id)
                    for ce_loss in self.CE_losses:
                        ce_loss.cuda(self.gpu_id)

        # Load the dataset
        if not test_only:
            if self.dataset == 'mnist':
                X, Y = utils.load_mnist(args.dataset)
                dset = TensorDataset(X, Y)
                self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)

            elif self.dataset == '3Dchairs':
                trans = transforms.Compose([transforms.Scale(self.im_resize), transforms.Grayscale(), transforms.ToTensor()])
                self.data_loader = utils.load_3Dchairs(transform=trans, batch_size=self.batch_size)

            elif self.dataset == 'synth':
                trans = transforms.Compose([transforms.Scale(self.im_resize), transforms.Grayscale(), transforms.ToTensor()])
                self.data_loader = utils.load_synth(transform=trans, batch_size=self.batch_size)

            # Creates train history dictionnary to record important training indicators
            self.train_history = {}
            self.train_history['D_loss'] = []
            self.train_history['G_loss'] = []
            self.train_history['per_epoch_time'] = []
            self.train_history['total_time'] = []
            if is_infogan:
                self.train_history['info_loss'] = []

    def save(self):
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_type + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, self.model_type + '_D.pkl'))

        with open(os.path.join(self.save_dir, self.model_type + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_history, f)

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_type + '_G.pkl')))
        if not self.test_only:
            self.D.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_type + '_D.pkl')))