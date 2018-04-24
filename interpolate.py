import argparse, os
from GAN import GAN
from WGAN import WGAN

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str,
                        choices=['GAN', 'WGAN'],
                        help='The type of GAN')#, required=True)
    
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')
    
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    
    parser.add_argument('--result_dir', type=str, default='interpolations',
                        help='Directory name to save the interpolated images')
    

    # UNUSED
    parser.add_argument('--dataset', type=str, default='celebA', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=False)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def generate_9_samples(gan):
	seeds = [1234, 5678, 4321, 9876, 1357]
	
	for seed in seeds:
		torch.manual_seed(seed)
		# Creates save dir
		save_dir = os.path.join(gan.result_dir, gan.gan_type, '9samples')
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# Creates 9 samples and saves them
		plt.figure(figsize=(20,20))
		for i in range(9):
			z = Variable(torch.rand((gan.batch_size, gan.z_dim)), volatile=True)
			x = gan.G(z)
			x = x.data.numpy().transpose(0, 2, 3, 1).squeeze()

			plt.subplot(3,3,i+1)
			plt.imshow(x)
			plt.axis('off')

		plt.savefig(os.path.join(save_dir, 'seed{}.png'.format(seed)), bbox_inches='tight')

def make_changes_in_latent_space(gan):
	changes = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
	seed = 11223344
	
	# Creates save dir
	save_dir = os.path.join(gan.result_dir, gan.gan_type, 'latentSpace')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	plt.figure(figsize=(20,6))
	for i in range(3):
		for j, change in enumerate(changes):
			torch.manual_seed(seed)
			z = Variable(torch.rand((gan.batch_size, gan.z_dim)), volatile=True)

			z[0,i] = z[0,i] + change
			x = gan.G(z)
			x = x.data.numpy().transpose(0, 2, 3, 1).squeeze()

			plt.subplot(3,len(changes),(len(changes)*i)+j+1)
			plt.imshow(x)

			if change == 0:
				plt.title('Sample $z$')
			if change > 0:
				plt.title('$z_{0}$ $\leftarrow$ $z_{0}$ + {1}'.format(i+1, abs(change)))
			if change < 0:
				plt.title('$z_{0}$ $\leftarrow$ $z_{0}$ - {1}'.format(i+1, abs(change)))
			plt.axis('off')

	plt.savefig(os.path.join(save_dir, 'seed{}.png'.format(seed)), bbox_inches='tight')


def interpolate_between_two_points(gan):
	seed_0 = 4444
	seed_1 = 11223344

	torch.manual_seed(seed_0)
	z_0 = Variable(torch.rand((gan.batch_size, gan.z_dim)), volatile=True)

	torch.manual_seed(seed_1)
	z_1 = Variable(torch.rand((gan.batch_size, gan.z_dim)), volatile=True)

	x_0 = gan.G(z_0)
	x_0 = x_0.data.numpy().transpose(0, 2, 3, 1).squeeze()

	x_1 = gan.G(z_1)
	x_1 = x_1.data.numpy().transpose(0, 2, 3, 1).squeeze()

	# Creates save dir
	save_dir = os.path.join(gan.result_dir, gan.gan_type, 'interpolations')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	alphas = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
	plt.figure(figsize=(20,4))

	# Interpolation in latent space
	for i, a in enumerate(alphas):
		z = a*z_0 + (1 - a)*z_1

		x = gan.G(z)
		x = x.data.numpy().transpose(0, 2, 3, 1).squeeze()

		plt.subplot(2, 11, i+1)
		plt.imshow(x)
		plt.axis('off')
		plt.title(r'$\alpha$ = {}'.format(a))

	# Interpolation in pixel space
	for i, a in enumerate(alphas):
		x = a*x_0 + (1 - a)*x_1

		plt.subplot(2, 11, 11+i+1)
		plt.imshow(x)
		plt.axis('off')

	plt.savefig(os.path.join(save_dir, 'seeds{}_{}.png'.format(seed_0, seed_1)), bbox_inches='tight')



"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args, test_only=True)
    elif args.gan_type == 'WGAN':
        gan = WGAN(args, test_only=True)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

    # Loads the model
    gan.load()
    print('[*] Model sucessfully loaded')

    # Creates samples of the model
    gan.G.eval()

    # Applies the operations asked in question 5 a) b) and c)
    generate_9_samples(gan)
    make_changes_in_latent_space(gan)
    interpolate_between_two_points(gan)

if __name__ == '__main__':
    main()