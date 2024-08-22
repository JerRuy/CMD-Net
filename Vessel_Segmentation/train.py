from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from scripts.loss import  *
from scripts.data import ImageFolder
# from scripts.quant import *


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "1"




def CMD_Net_Train(net, parameter_mask, old_weight_dict ,input_size,  image_path,  save_path,  alpha, epoch, batchsize, lr, NUM_EARLY_STOP, NUM_UPDATE_LR, INITAL_EPOCH_LOSS):

    print(epoch)
    NAME = 'CMD-Net_' + image_path.split('/')[-1]
    no_optim = 0
    total_epoch = epoch
    train_epoch_best_loss = INITAL_EPOCH_LOSS
    batchsize = batchsize

    dataset = ImageFolder(root_path=image_path, datasets='')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss = dice_bce_loss()


    tic = time()
    for epoch in range(1, total_epoch + 1):

        train_epoch_segmentation_loss = 0
        data_loader_iter = iter(data_loader)

        for img, mask in data_loader_iter:
            cv2.imwrite('image.jpg',img.numpy())
            cv2.imwrite('label.jpg', mask.numpy())
            img = V(img.cuda(), volatile=False)
            mask = V(mask.cuda(), volatile=False)
            optimizer.zero_grad()
            pred, _ = net.forward(img)
            # pred = pred.clamp(0, 1)
            train_loss = loss(mask, pred)
            train_loss.backward()
            optimizer.step()
            train_epoch_segmentation_loss += train_loss

        train_epoch_segmentation_loss = train_epoch_segmentation_loss / len(data_loader_iter)
        print(' epoch: ', epoch, ' time:', int(time() - tic), ' loss: ', round(train_epoch_segmentation_loss.item(), 6))

        if save_path.endswith('/') is False:
            save_path = save_path + "/"

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if train_epoch_segmentation_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_segmentation_loss
            torch.save(net.state_dict(), save_path + NAME + '.th')
        if no_optim > NUM_EARLY_STOP:
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > NUM_UPDATE_LR:
            if lr < 5e-7:
                break
            net.load_state_dict(torch.load(save_path + NAME + '.th'))
            lr = lr / 2

    print('Finish!')
    return net


def loadweight(net, path):
    model_dict = net.state_dict()
    checkpoint = torch.load(path)

    for k, v in checkpoint.items():
        name = k
        model_dict[name] = v
    net.load_state_dict(model_dict)
    return net

