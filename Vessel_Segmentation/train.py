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
from scripts.quant import *


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


# if __name__ == '__main__':
#     print(torch.__version__)

#     parser = ArgumentParser(description="Evaluation script for CMD-Net models",formatter_class=ArgumentDefaultsHelpFormatter)

#     parser.add_argument('--input-size', default=448, type=int, help='Images input size')
#     parser.add_argument('--image-path', default='./dataset/DRIVE', type=str, help='DIRVE dataset path')
#     parser.add_argument('--model', default='FCNsa', type=str, help='train model')
#     parser.add_argument('--save-path', default='weights/1/', type=str, help='store the trained model')
#     parser.add_argument('--alpha', default=0.0001, type=float, help='the empirical coefficient for diversity')
#     parser.add_argument('--epoch', default=300, type=int, help='')
#     parser.add_argument('--batchsize', default=4, type=int, help='Batch per GPU')
#     parser.add_argument('--n-channels', default=3, type=int, help='n channels')
#     parser.add_argument('--n-classes', default=1, type=int, help='classes')
#     parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
#     parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
#     parser.add_argument('--early-stop', default=20, type=int, help=' ')
#     parser.add_argument('--update-lr', default=10, type=int, help=' ')
#     parser.add_argument('--epochloss_init', default=10000, type=int, help=' ')

#     args = parser.parse_args()

#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

#     if (args.model == 'FCN'):
#         net = FCN(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'UNet'):
#         net = UNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'FCNsa'):
#         net = FCNsa(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'DeepLab'):
#         net = DeepLabV3(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'SegNet'):
#         net = SegNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'ConvUNeXt'):
#         net = ConvUNeXt(in_channels = args.n_channels, num_classes = args.n_classes).cuda()
#     elif (args.model == 'CMDNet_FCN'):
#         net = CMDNet_FCN(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'CMDNet_FCNsa'):
#         net = CMDNet_FCNsa(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'CMDNet_UNet'):
#         net = CMDNet_UNet(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'CMDNet_SegNet'):
#         net = CMDNet_SegNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'CMDNet_DeepLab'):
#         net = CMDNet_DeepLab(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
#     elif (args.model == 'CMDNet_ConvUNeXt'):
#         net = CMDNet_ConvUNeXt(in_channels = args.n_channels, num_classes = args.n_classes).cuda()
#     else:
#         print("the model name must be in CMDNet_FCN, CMDNet_FCNsa, CMDNet_UNet, CMDNet_ConvUNeXt.")


#     net = CMD_Net_Train(net, [], [],  args.input_size,  args.image_path,  args.save_path, args.alpha, args.epoch, args.batchsize, args.lr,  args.early_stop, args.update_lr, args.epochloss_init )

