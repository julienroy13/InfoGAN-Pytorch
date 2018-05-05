import os
import time
import torch
from torch.autograd import Variable
import numpy as np
import utils

def train(model):
    # Checks what kind of model it is training
    if model.model_type == "infoGAN":
        is_infogan = True
    else:
        is_infogan = False

    # Makes sure we have a dir to save the model and training info
    if not os.path.exists(model.save_dir):
        os.makedirs(model.save_dir)

    # Creates artificial labels that just indicates to the loss object if prediction of D should be 0 or 1
    if model.gpu_mode:
        y_real_ = Variable(torch.ones(model.batch_size, 1).cuda(model.gpu_id)) # all ones
        y_fake_ = Variable(torch.zeros(model.batch_size, 1).cuda(model.gpu_id)) # all zeros
    else:
        y_real_ = Variable(torch.ones(model.batch_size, 1))
        y_fake_ = Variable(torch.zeros(model.batch_size, 1))

    model.D.train() # sets discriminator in train mode

    # TRAINING LOOP
    start_time = time.time()
    print('[*] TRAINING STARTS')
    for epoch in range(model.epoch):
        model.G.train() # sets generator in train mode
        epoch_start_time = time.time()
        
        # For each minibatch returned by the data_loader
        for step, (x_, _) in enumerate(model.data_loader):
            if step == model.data_loader.dataset.__len__() // model.batch_size:
                break

            # Creates a minibatch of latent vectors
            z_ = torch.rand((model.batch_size, model.z_dim))

            # Creates a minibatch of discrete and continuous codes
            c_disc_ = torch.from_numpy(np.random.multinomial(1, model.c_disc_dim * [float(1.0 / model.c_disc_dim)], size=[model.batch_size])).type(torch.FloatTensor)
            for i in range(model.n_disc_code-1):
                c_disc_ = torch.cat([c_disc_, torch.from_numpy(np.random.multinomial(1, model.c_disc_dim * [float(1.0 / model.c_disc_dim)], size=[model.batch_size])).type(torch.FloatTensor)], dim=1)
            c_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(model.batch_size, model.c_cont_dim))).type(torch.FloatTensor)

            # Convert to Variables (sends to GPU if needed)
            if model.gpu_mode:
                x_ = Variable(x_.cuda(model.gpu_id))
                z_ = Variable(z_.cuda(model.gpu_id))
                c_disc_ = Variable(c_disc_.cuda(model.gpu_id))
                c_cont_ = Variable(c_cont_.cuda(model.gpu_id))
            else:
                x_ = Variable(x_)
                z_ = Variable(z_)
                c_disc_ = Variable(c_disc_)
                c_cont_ = Variable(c_cont_)

            # update D network
            model.D_optimizer.zero_grad()

            D_real, _, _ = model.D(x_)
            D_real_loss = model.BCE_loss(D_real, y_real_)

            G_ = model.G(z_, c_cont_, c_disc_)
            D_fake, _, _ = model.D(G_)
            D_fake_loss = model.BCE_loss(D_fake, y_fake_)

            D_loss = D_real_loss + D_fake_loss
            model.train_history['D_loss'].append(D_loss.data[0])

            D_loss.backward(retain_graph=True)
            model.D_optimizer.step()

            # update G network
            model.G_optimizer.zero_grad()

            G_ = model.G(z_, c_cont_, c_disc_)
            D_fake, D_cont, D_disc = model.D(G_)

            G_loss = model.BCE_loss(D_fake, y_real_)
            model.train_history['G_loss'].append(G_loss.data[0])

            G_loss.backward(retain_graph=True)
            model.G_optimizer.step()

            # information loss
            if is_infogan:
                disc_loss = 0
                for ce_loss in model.CE_losses:
                    disc_loss += ce_loss(D_disc, torch.max(c_disc_, 1)[1])
                cont_loss = model.MSE_loss(D_cont, c_cont_)
                info_loss = disc_loss + cont_loss
                model.train_history['info_loss'].append(info_loss.data[0])

                info_loss.backward()
                model.info_optimizer.step()

            # Prints training info every 100 steps
            if ((step + 1) % 100) == 0:
                if is_infogan:
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}, info_loss: {:.8f}".format(
                          (epoch + 1), (step + 1), model.data_loader.dataset.__len__() // model.batch_size, D_loss.data[0], G_loss.data[0], info_loss.data[0]))
                else:
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}".format(
                              (epoch + 1), (step + 1), model.data_loader.dataset.__len__() // model.batch_size, D_loss.data[0], G_loss.data[0]))

        model.train_history['per_epoch_time'].append(time.time() - epoch_start_time)

        # Saves samples
        utils.generate_samples(model, os.path.join(model.save_dir, "epoch{}.png".format(epoch)))

    model.train_history['total_time'].append(time.time() - start_time)
    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(model.train_history['per_epoch_time']),
          model.epoch, model.train_history['total_time'][0]))
    print("[*] TRAINING FINISHED")

    # Saves the model
    model.save()
    
    # Saves the plot of losses for G and D
    utils.save_loss_plot(model.train_history, filename=os.path.join(model.save_dir, "curves.png"), infogan=is_infogan)