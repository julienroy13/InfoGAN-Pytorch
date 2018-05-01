# code inspired by : github.com/znxlwm/pytorch-generative-model-collections
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
        x = torch.cat([z, cont_code, discr_code], 1)
        
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
        b = x[:, 1:1+self.len_continuous_code] # continuous codes
        c = x[:, 1+self.len_continuous_code:]  # discrete codes

        return a, b, c

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()

class infoGAN(object):
    def __init__(self, args, test_only=False):
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.dataset = args.dataset
        self.gpu_mode = args.gpu_mode
        self.gpu_id = args.gpu_id
        self.model_name = args.gan_type

        # Defines input/output dimensions
        if self.dataset == 'mnist':
            self.x_height = 28
            self.x_width = 28
            self.x_features = 1
            self.y_dim = 1
            self.c_discr_dim = 10     # categorical distribution (i.e. label)
            self.c_cont_dim = 2       # gaussian distribution (e.g. rotation, thickness)
            self.c_dim = self.c_discr_dim + self.c_cont_dim
            self.z_dim = 62

        elif dataset == '3Dchairs':
            raise NotImplemented

        elif dataset == 'synth':
            raise NotImplemented

        else:
            raise Exception('Unsupported dataset')

        # Initializes the models and their optimizers
        self.G = generator(self.z_dim + self.c_dim, self.x_height, self.x_width, self.x_features)
        utils.print_network(self.G)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        
        self.D = discriminator(self.x_height, self.x_width, self.x_features, self.y_dim, self.c_discr_dim, self.c_cont_dim)
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
                self.data_X, self.data_Y = utils.load_mnist(args.dataset)
                #dset = TensorDataset(X, Y)
                #self.data_loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(10):
            self.sample_z_[i*self.c_discr_dim] = torch.rand(1, self.z_dim)
            for j in range(1, self.c_discr_dim):
                self.sample_z_[i*self.c_discr_dim + j] = self.sample_z_[i*self.c_discr_dim]

        temp = torch.zeros((10, 1))
        for i in range(self.c_discr_dim):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(10):
            temp_y[i*self.c_discr_dim: (i+1)*self.c_discr_dim] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.c_discr_dim))
        self.sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)
        self.sample_c_ = torch.zeros((self.sample_num, self.len_continuous_code))

        # manipulating two continuous code
        temp_z_ = torch.rand((1, self.z_dim))
        self.sample_z2_ = temp_z_
        for i in range(self.sample_num - 1):
            self.sample_z2_ = torch.cat([self.sample_z2_, temp_z_])

        y = np.zeros(self.sample_num, dtype=np.int64)
        y_one_hot = np.zeros((self.sample_num, self.len_discrete_code))
        y_one_hot[np.arange(self.sample_num), y] = 1
        self.sample_y2_ = torch.from_numpy(y_one_hot).type(torch.FloatTensor)

        temp_c = torch.linspace(-1, 1, 10)
        self.sample_c2_ = torch.zeros((self.sample_num, 2))
        for i in range(10):
            for j in range(10):
                self.sample_c2_[i*10+j, 0] = temp_c[i]
                self.sample_c2_[i*10+j, 1] = temp_c[j]

        if self.gpu_mode:
            self.sample_z_, self.sample_y_, self.sample_c_, self.sample_z2_, self.sample_y2_, self.sample_c2_ = \
                Variable(self.sample_z_.cuda(self.gpu_id), volatile=True), Variable(self.sample_y_.cuda(self.gpu_id), volatile=True), \
                Variable(self.sample_c_.cuda(self.gpu_id), volatile=True), Variable(self.sample_z2_.cuda(self.gpu_id), volatile=True), \
                Variable(self.sample_y2_.cuda(self.gpu_id), volatile=True), Variable(self.sample_c2_.cuda(self.gpu_id), volatile=True)
        else:
            self.sample_z_, self.sample_y_, self.sample_c_, self.sample_z2_, self.sample_y2_, self.sample_c2_ = \
                Variable(self.sample_z_, volatile=True), Variable(self.sample_y_, volatile=True), \
                Variable(self.sample_c_, volatile=True), Variable(self.sample_z2_, volatile=True), \
                Variable(self.sample_y2_, volatile=True), Variable(self.sample_c2_, volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['info_loss'] = []
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
            for iter in range(len(self.data_X) // self.batch_size):
                x_ = self.data_X[iter*self.batch_size:(iter+1)*self.batch_size]
                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.SUPERVISED == True:
                    y_disc_ = self.data_Y[iter*self.batch_size:(iter+1)*self.batch_size]
                else:
                    y_disc_ = torch.from_numpy(
                        np.random.multinomial(1, self.len_discrete_code * [float(1.0 / self.len_discrete_code)],
                                              size=[self.batch_size])).type(torch.FloatTensor)

                y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size, 2))).type(torch.FloatTensor)

                if self.gpu_mode:
                    x_, z_, y_disc_, y_cont_ = Variable(x_.cuda(self.gpu_id)), Variable(z_.cuda(self.gpu_id)), \
                                               Variable(y_disc_.cuda(self.gpu_id)), Variable(y_cont_.cuda(self.gpu_id))
                else:
                    x_, z_, y_disc_, y_cont_ = Variable(x_), Variable(z_), Variable(y_disc_), Variable(y_cont_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real, _, _ = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_cont_, y_disc_)
                D_fake, _, _ = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_cont_, y_disc_)
                D_fake, D_cont, D_disc = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                # information loss
                disc_loss = self.CE_loss(D_disc, torch.max(y_disc_, 1)[1])
                cont_loss = self.MSE_loss(D_cont, y_cont_)
                info_loss = disc_loss + cont_loss
                self.train_hist['info_loss'].append(info_loss.data[0])

                info_loss.backward()
                self.info_optimizer.step()


                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.data_X) // self.batch_size, D_loss.data[0], G_loss.data[0], info_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_cont',
                                 self.epoch)
        self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        """ style by class """
        samples = self.G(self.sample_z_, self.sample_c_, self.sample_y_)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

        """ manipulating two continous codes """
        samples = self.G(self.sample_z2_, self.sample_c2_, self.sample_y2_)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_cont_epoch%03d' % epoch + '.png')

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
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        y3 = hist['info_loss']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')
        plt.plot(x, y3, label='info_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)