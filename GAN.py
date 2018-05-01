# code inspired by : github.com/znxlwm/pytorch-generative-model-collections
import utils
import torch
import time
import os
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

class generator(nn.Module):
    def __init__(self, latent_dim, output_height, output_width, output_features):
        super(generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_height = output_height
        self.output_width = output_width
        self.output_features = output_features

        # First layers are fully connected
        self.fc_part = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.output_height // 4) * (self.output_width // 4)),
            nn.BatchNorm1d(128 * (self.output_height // 4) * (self.output_width // 4)),
            nn.ReLU(),
        )

        # Then we switch to deconvolution (transpose convolutions)
        self.deconv_part = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_features, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(), #[0, 1] images
        )
        
        self.initialize_weights()

    def forward(self, z):
        # Forwards through first fully connected layers
        x = self.fc_part(z)
        
        # Reshapes into feature maps 4 times smaller than original size
        x = x.view(-1, 128, (self.output_height // 4), (self.output_width // 4))
        
        # Feedforward through deconvolutional part (upsampling)
        x = self.deconv_part(x)

        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()

class discriminator(nn.Module):
    def __init__(self, input_height, input_width, input_features, output_dim):
        super(discriminator, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_features = input_features
        self.output_dim = output_dim

        # First layers are convolutional (subsampling)
        self.conv_part = nn.Sequential(
            nn.Conv2d(self.input_features, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Then we switch to fully connected before sigmoidal output unit
        self.fc_part = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        self.initialize_weights()

    def forward(self, x):
        # Feedforwards through convolutional (subsampling) layers
        y = self.conv_part(x)
        
        # Reshapes as a vector
        y = y.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        
        # Feedforwards through fully connected layers
        y = self.fc_part(y)

        return y

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()


class GAN(object):
    def __init__(self, args, test_only=False):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.dataset = args.dataset
        self.gpu_mode = args.gpu_mode
        self.gpu_id = args.gpu_id
        self.test_only = test_only
        self.z_dim = 62

        # Defines input/output dimensions
        if self.dataset == 'mnist':
            self.x_height = 28
            self.x_width = 28
            self.x_features = 1
            self.y_dim = 1
            self.z_dim = 62

        elif dataset == '3Dchairs':
            raise NotImplemented

        elif dataset == 'synth':
            raise NotImplemented

        else:
            raise Exception('Unsupported dataset')

        # Initializes the models and their optimizers
        self.G = generator(self.z_dim, self.x_height, self.x_width, self.x_features)
        utils.print_network(self.G)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        
        if not test_only: 
            self.D = discriminator(self.x_height, self.x_width, self.x_features, self.y_dim)
            utils.print_network(self.D)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # Loss function
        self.BCE_loss = nn.BCELoss()

        # Sends the models of GPU (if defined)
        if self.gpu_mode:
            self.G.cuda(self.gpu_id)
            if not test_only: self.D.cuda(self.gpu_id)
            self.BCE_loss.cuda(self.gpu_id)            

        # Load the dataset
        if not test_only:
            if self.dataset == 'mnist':
                X, Y = utils.load_mnist(args.dataset)
                dset = TensorDataset(X, Y)
                self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)

        # Creates the random latent vectors
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(self.gpu_id), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

        # Creates train history dictionnary to record important training indicators
        self.train_history = {}
        self.train_history['D_loss'] = []
        self.train_history['G_loss'] = []
        self.train_history['per_epoch_time'] = []
        self.train_history['total_time'] = []

    def train(self):

        # Creates artificial labels that just indicates to the loss object if prediction of D should be 0 or 1
        if self.gpu_mode:
            self.y_real_ = Variable(torch.ones(self.batch_size, 1).cuda(self.gpu_id)) # all ones
            self.y_fake_ = Variable(torch.zeros(self.batch_size, 1).cuda(self.gpu_id)) # all zeros
        else:
            self.y_real_ =  Variable(torch.ones(self.batch_size, 1))
            self.y_fake_ = Variable(torch.zeros(self.batch_size, 1))

        self.D.train() # sets discriminator in train mode
        
        # TRAINING LOOP
        start_time = time.time()
        print('[*] TRAINING STARTS')
        for epoch in range(self.epoch):
            self.G.train() # sets generator in train mode
            epoch_start_time = time.time()
            
            # For each minibatch returned by the data_loader
            for step, (x_, _) in enumerate(self.data_loader):
                if step == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                # Creates a minibatch of latent vectors
                z_ = torch.rand((self.batch_size, self.z_dim))

                # Convert to Variables (sends to GPU if needed)
                if self.gpu_mode:
                    x_ = Variable(x_.cuda(self.gpu_id))
                    z_ = Variable(z_.cuda(self.gpu_id))
                else:
                    x_ = Variable(x_) 
                    z_ = Variable(z_)

                # Updates D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_history['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # Updates G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_history['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                # Prints training info every 100 steps
                if ((step + 1) % 100) == 0:
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}".format(
                          (epoch + 1), (step + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.data[0], G_loss.data[0]))

            self.train_history['per_epoch_time'].append(time.time() - epoch_start_time)

        self.train_history['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_history['per_epoch_time']),
              self.epoch, self.train_history['total_time'][0]))
        print("[*] TRAINING FINISHED")

        # Saves the model
        self.save()
        
        # Saves the plot of losses for G and D
        utils.save_loss_plot(self.train_history, filename=os.path.join(self.save_dir, self.dataset, "GAN", 'curves'))

    def save(self):
        # Defines save directory
        save_dir = os.path.join(self.save_dir, self.dataset, "GAN")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Saves the models
        torch.save(self.G.state_dict(), os.path.join(save_dir, "GAN" + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, "GAN" + '_D.pkl'))

        # Saves the train history
        with open(os.path.join(save_dir, "GAN" + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_history, f)

    def load(self):
        # Defines save directory
        save_dir = os.path.join(self.save_dir, self.dataset, "GAN")

        # Loads the necessary models
        self.G.load_state_dict(torch.load(os.path.join(save_dir, "GAN" + '_G.pkl')))
        if not self.test_only: 
            self.D.load_state_dict(torch.load(os.path.join(save_dir, "GAN" + '_D.pkl')))