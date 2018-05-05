import utils
import torch
import time
import os
import pickle
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

    def forward(self, z, cont_code, discr_code):
        # Concatenates latent vector and latent codes (continuous and discrete)
        x = torch.cat([z, cont_code, discr_code], dim=1)
        
        # Forwards through first fully connected layers
        x = self.fc_part(x)
        
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
    def __init__(self, input_height, input_width, input_features, output_dim, len_discrete_code, len_continuous_code):
        super(discriminator, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_features = input_features
        self.output_dim = output_dim
        self.len_discrete_code = len_discrete_code
        self.len_continuous_code = len_continuous_code

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
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
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

        # D output
        a = F.sigmoid(y[:, 0]) # MODIF

        # Q outputs
        b = y[:, 1:1+self.len_continuous_code] # continuous codes
        c = y[:, 1+self.len_continuous_code:]  # discrete codes

        return a, b, c

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()

class infoGAN(object):
    def __init__(self, args, test_only=False):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.save_dir = os.path.join(args.save_dir, self.dataset, "infoGAN")
        self.gpu_mode = args.gpu_mode
        self.gpu_id = args.gpu_id

        # Defines input/output dimensions
        if self.dataset == 'mnist':
            self.x_height = 28
            self.x_width = 28
            self.x_features = 1
            self.y_dim = 1
            self.c_disc_dim = 10 
            self.c_cont_dim = 2   
            self.c_dim = self.c_disc_dim + self.c_cont_dim
            self.z_dim = 62

        elif self.dataset == '3Dchairs':
            self.im_resize = 128
            self.x_height = self.im_resize
            self.x_width = self.im_resize
            self.x_features = 3
            self.y_dim = 1
            self.c_disc_dim = 20
            self.c_cont_dim = 1
            self.c_dim = self.c_disc_dim + self.c_cont_dim
            self.z_dim = 62

        elif self.dataset == 'synth':
            self.im_resize = 128
            self.x_height = self.im_resize
            self.x_width = self.im_resize
            self.x_features = 1
            self.y_dim = 1
            self.c_disc_dim = 10
            self.c_cont_dim = 3
            self.c_dim = self.c_disc_dim + self.c_cont_dim
            self.z_dim = 62

        else:
            raise Exception('Unsupported dataset')

        # Initializes the models and their optimizers
        self.G = generator(self.z_dim + self.c_dim, self.x_height, self.x_width, self.x_features)
        utils.print_network(self.G)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        
        self.D = discriminator(self.x_height, self.x_width, self.x_features, self.y_dim, self.c_disc_dim, self.c_cont_dim)
        utils.print_network(self.D)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        
        self.info_optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        # Loss functions
        self.BCE_loss = nn.BCELoss()
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()

        # Sends the models of GPU (if defined)
        if self.gpu_mode:
            self.G.cuda(self.gpu_id)
            self.D.cuda(self.gpu_id)
            self.BCE_loss.cuda(self.gpu_id)
            self.CE_loss.cuda(self.gpu_id)
            self.MSE_loss.cuda(self.gpu_id)

        # Load the dataset
        if not test_only:
            if self.dataset == 'mnist':
                X, Y = utils.load_mnist(args.dataset)
                dset = TensorDataset(X, Y)
                self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)

            elif self.dataset == '3Dchairs':
                trans = transforms.Compose([transforms.Scale(self.im_resize), transforms.ToTensor()])
                self.data_loader = utils.load_3Dchairs(transform=trans, batch_size=self.batch_size)

            elif self.dataset == 'synth':
                trans = transforms.Compose([transforms.Scale(self.im_resize), transforms.Grayscale(), transforms.ToTensor()])
                self.data_loader = utils.load_synth(transform=trans, batch_size=self.batch_size)

        # Creates train history dictionnary to record important training indicators
        self.train_history = {}
        self.train_history['D_loss'] = []
        self.train_history['G_loss'] = []
        self.train_history['info_loss'] = []
        self.train_history['per_epoch_time'] = []
        self.train_history['total_time'] = []

    def train(self):
        # Makes sure we have a dir to save the model and training info
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Creates artificial labels that just indicates to the loss object if prediction of D should be 0 or 1
        if self.gpu_mode:
            self.y_real_ = Variable(torch.ones(self.batch_size, 1).cuda(self.gpu_id)) # all ones
            self.y_fake_ = Variable(torch.zeros(self.batch_size, 1).cuda(self.gpu_id)) # all zeros
        else:
            self.y_real_ = Variable(torch.ones(self.batch_size, 1))
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

                # Creates a minibatch of discrete and continuous codes
                c_disc_ = torch.from_numpy(np.random.multinomial(1, self.c_disc_dim * [float(1.0 / self.c_disc_dim)], size=[self.batch_size])).type(torch.FloatTensor)
                c_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size, self.c_cont_dim))).type(torch.FloatTensor)

                # Convert to Variables (sends to GPU if needed)
                if self.gpu_mode:
                    x_ = Variable(x_.cuda(self.gpu_id))
                    z_ = Variable(z_.cuda(self.gpu_id))
                    c_disc_ = Variable(c_disc_.cuda(self.gpu_id))
                    c_cont_ = Variable(c_cont_.cuda(self.gpu_id))
                else:
                    x_ = Variable(x_)
                    z_ = Variable(z_)
                    c_disc_ = Variable(c_disc_)
                    c_cont_ = Variable(c_cont_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real, _, _ = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, c_cont_, c_disc_)
                D_fake, _, _ = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_history['D_loss'].append(D_loss.data[0])

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, c_cont_, c_disc_)
                D_fake, D_cont, D_disc = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_history['G_loss'].append(G_loss.data[0])

                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                # information loss
                disc_loss = self.CE_loss(D_disc, torch.max(c_disc_, 1)[1])
                cont_loss = self.MSE_loss(D_cont, c_cont_)
                info_loss = disc_loss + cont_loss
                self.train_history['info_loss'].append(info_loss.data[0])

                info_loss.backward()
                self.info_optimizer.step()

                # Prints training info every 100 steps
                if ((step + 1) % 100) == 0:
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}, info_loss: {:.8f}".format(
                          (epoch + 1), (step + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.data[0], G_loss.data[0], info_loss.data[0]))

            self.train_history['per_epoch_time'].append(time.time() - epoch_start_time)

            # Saves samples
            utils.generate_samples(self, self.z_dim, os.path.join(self.save_dir, "epoch{}.png".format(epoch)), self.c_cont_dim, self.c_disc_dim)

        self.train_history['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_history['per_epoch_time']),
              self.epoch, self.train_history['total_time'][0]))
        print("[*] TRAINING FINISHED")

        # Saves the model
        self.save()
        
        # Saves the plot of losses for G and D
        utils.save_loss_plot(self.train_history, filename=os.path.join(self.save_dir, "curves.png"), infogan=True)
        

    def save(self):
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, "infoGAN" + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, "infoGAN" + '_D.pkl'))

        with open(os.path.join(self.save_dir, "infoGAN" + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_history, f)

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.save_dir, "infoGAN" + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.save_dir, "infoGAN" + '_D.pkl')))