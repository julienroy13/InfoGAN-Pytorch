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
    def __init__(self, dataset):
        super(generator, self).__init__()
        if dataset == 'mnist':
            self.latent_dim = 62
            self.output_height = 28
            self.output_width = 28
            self.output_features = 1

        elif dataset == '3Dchairs':
            raise NotImplemented

        elif dataset == 'synth':
            raise NotImplemented

        else:
            raise Exception('Unsupported dataset')

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

    def forward(self, input):
        # Forwards through first fully connected layers
        x = self.fc_part(input)
        
        # Reshapes into feature maps 4 times smaller than original size
        x = x.view(-1, 128, (self.output_height // 4), (self.output_width // 4))
        
        # Feedforward through deconvolutional part (upsampling)
        x = self.deconv_part(x)

        return x

    def initialize_weights(self):
        for module in self.modules():
            module.weight.data.normal_(0, 0.02)
            module.bias.data.zero_()

class discriminator(nn.Module):
    def __init__(self, dataset='mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_features = 1
            self.output_dim = 1

        elif dataset == '3Dchairs':
            raise NotImplemented

        elif dataset == 'synth':
            raise NotImplemented

        else:
            raise Exception('Unsupported dataset')

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

    def forward(self, input):
        # Feedforwards through convolutional (subsampling) layers
        x = self.conv_part(input)
        
        # Reshapes as a vector
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        
        # Feedforwards through fully connected layers
        x = self.fc_part(x)

        return x

    def initialize_weights(self):
        for module in self.modules():
            module.weight.data.normal_(0, 0.02)
            module.bias.data.zero_()


class GAN(object):
    def __init__(self, args, test_only=False):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 16
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.gpu_id = args.gpu_id
        self.model_name = args.gan_type
        self.test_only = test_only
        self.gan_type = args.gan_type
        self.z_dim = 62

        # Initializes the models and their optimizers
        self.G = generator(self.dataset)
        utils.print_network(self.G)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        if not test_only: 
            self.D = discriminator(self.dataset)
            utils.print_network(self.D)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # Loss function
        self.BCE_loss = nn.BCELoss()

        # Sends the models of GPU (if defined)
        if self.gpu_mode:
            self.G.cuda(self.gpu_id)
            if not test_only: self.D.cuda(self.gpu_id)
            nn.BCELoss().cuda(self.gpu_id)            

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

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.gpu_id)), Variable(torch.zeros(self.batch_size, 1).cuda(self.gpu_id))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda(self.gpu_id)), Variable(z_.cuda(self.gpu_id))
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(self.gpu_id), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        if not self.test_only: self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))