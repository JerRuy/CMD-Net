import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import sklearn.metrics as metrics
import cv2
import os
import sys
import numpy as np

from time import time
from PIL import Image

import warnings

warnings.filterwarnings('ignore')

# from networks.cenet_cq3_ca import CE_Net_T
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from networks.UNet import UNet
from networks.FCN import FCN
from networks.FCNsa import FCNsa
from networks.DeepLab import DeepLabV3
from networks.SegNet import SegNet
from networks.ConvUNeXt import ConvUNeXt
from networks.CMDNet_FCN import CMDNet_FCN
from networks.CMDNet_FCNsa import CMDNet_FCNsa
from networks.CMDNet_UNet import CMDNet_UNet
from networks.CMDNet_SegNet import CMDNet_SegNet
from networks.CMDNet_ConvUNeXt import CMDNet_ConvUNeXt
from networks.CMDNet_DeepLab import CMDNet_DeepLab

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BATCHSIZE_PER_CARD = 1


def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()

    # label_1D = label_1D / 255

    auc = metrics.roc_auc_score(label_1D, result_1D)

    # print("AUC={0:.4f}".format(auc))

    return auc


def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    # sen = TP / (TP + FN)
    # spec = TP / (TP + FP)

    Recall = TP / (TP + FN)
    # Precision = TP / (TP + FP)
    Precision = TN / (TN + FP)
    F1score = 2 * Precision * Recall / (Precision + Recall)

    # IoU = TP / (FN + TP + FP)
    vessel_IoU = TP / (FN + TP + FP)
    background_IoU = TN / (FN + TP + TN)
    IoU = (vessel_IoU+background_IoU)/2
    Dice = 2 * TP / (2 * TP + FP + FN)
    # return acc, sen
    return acc, Recall, Precision, F1score, IoU, Dice


class TTAFrame():
    def __init__(self, net):
        self.net = net
            # .cuda()
        # self.net.eval()
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path_8a1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img1)
        maskb, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img2)
        maskc, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img3)
        maskd, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maskaa, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg1)
        maskbb, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg2)
        maskcc, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg3)
        maskdd, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

    def test_one_img_from_path_8a2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, maska, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img1)
        mask, maskb, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img2)
        mask, maskc, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img3)
        mask, maskd, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, maskaa, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg1)
        mask, maskbb, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg2)
        mask, maskcc, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg3)
        mask, maskdd, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

    def test_one_img_from_path_8a3(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, mask, maska, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img1)
        mask, mask, maskb, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img2)
        mask, mask, maskc, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img3)
        mask, mask, maskd, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, mask, maskaa, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg1)
        mask, mask, maskbb, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg2)
        mask, mask, maskcc, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg3)
        mask, mask, maskdd, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

    def test_one_img_from_path_8l(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska, _ = self.net.forward(img1)
        maskb, _ = self.net.forward(img2)
        maskc, _ = self.net.forward(img3)
        maskd, _ = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maskaa, _  = self.net.forward(imgg1)
        maskbb, _ = self.net.forward(imgg2)
        maskcc, _ = self.net.forward(imgg3)
        maskdd, _ = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

def test_assistant(name, mask, gt_root):
    # gt_root = './dataset/DRIVE/test/1st_manual'
    # gt_root = './dataset/CHASEDB1/test/labels1'
    threshold = 6
    disc = 20

    new_mask = mask.copy()
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

    ext = name.split('_')[0]  + '_manual1.gif' if 'DRIVE' in gt_root else name.split('.')[0] + '_1stHO.png'
    ground_truth_path = os.path.join(gt_root, ext)
    ground_truth = np.array(Image.open(ground_truth_path))

    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

    new_mask = cv2.resize(new_mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

    predi_mask = np.zeros(shape=np.shape(mask))
    predi_mask[mask > disc] = 1
    gt = np.zeros(shape=np.shape(ground_truth))
    gt[ground_truth > 0] = 1

    acc, Recall, Precision, F1score, IoU, Dice = accuracy(predi_mask[:, :, 0], gt)
    auc = calculate_auc_test(new_mask / 8., ground_truth)
    # auc=0
    return acc, Recall, Precision, F1score, IoU, Dice, auc


def loadweight(net, path, gpuid):
    model_dict = net.state_dict()
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda(gpuid))

    for k, v in checkpoint.items():
        # name = k[7:]  # remove `module.`
        name = k
        # print('net',total,name)
        # total += 1
        model_dict[name] = v
    net.load_state_dict(model_dict)
    return net

def test_ce_net_vessel(net, path, gt_path):
    val = os.listdir(path)
    solver = TTAFrame(net)
    # acc, Recall, Precision, F1score, IoU, Dice
    total_acc = []
    total_recall = []
    total_precision = []
    total_f1score = []
    total_iou = []
    total_dice = []
    total_auc = []

    for i, name in enumerate(val):
        image_path = os.path.join(path, name)
        mask = solver.test_one_img_from_path_8l(image_path)
        acc, Recall, Precision, F1score, IoU, Dice, auc = test_assistant(name, mask, gt_path)

        total_acc.append(acc)
        total_recall.append(Recall)
        total_precision.append(Precision)
        total_f1score.append(F1score)
        total_iou.append(IoU)
        total_dice.append(Dice)
        total_auc.append(auc)
        print(
        i + 1, ' Accuracy: ', acc, ' Recall: ', Recall, ' Precision: ', Precision, ' F1score: ', F1score, ' IoU: ', IoU,
        ' Dice: ', Dice, ' AUC: ', auc)
    print(
    ' Accuracy: ', np.mean(total_acc), ' Recall: ', np.mean(total_recall), ' Precision: ', np.mean(total_precision),
    ' F1score: ', np.mean(total_f1score), ' IoU: ', np.mean(total_iou), ' Dice: ', np.mean(total_dice), ' AUC: ',
    np.mean(total_auc))

def CMD_Net_VS_Test(net, model_path, test_path, gt_path, gpuid):
    net.eval()
    net = loadweight(net, model_path, gpuid)
    test_ce_net_vessel(net,  test_path, gt_path)

if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluation script for CMD-Net models",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', default='weights/1/NQ4/CMD-Net_DRIVE.th', type=str, help='store the trained model')
    parser.add_argument('--test-path', default='./dataset/DRIVE/test/images/', type=str, help='store the test image')
    parser.add_argument('--gt-path', default='./dataset/DRIVE/test/1st_manual', type=str, help='store the test label')
    parser.add_argument('--model', default='FCNsa', type=str, help='train model')
    parser.add_argument('--n-channels', default=3, type=int, help='n channels')
    parser.add_argument('--n-classes', default=1, type=int, help='classes')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)


    if (args.model == 'FCN'):
        net = FCN(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'UNet'):
        net = UNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'FCNsa'):
        net = FCNsa(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'DeepLab'):
        net = DeepLabV3(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'SegNet'):
        net = SegNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'ConvUNeXt'):
        net = ConvUNeXt(in_channels = args.n_channels, num_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_FCN'):
        net = CMDNet_FCN(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_FCNsa'):
        net = CMDNet_FCNsa(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_UNet'):
        net = CMDNet_UNet(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_SegNet'):
        net = CMDNet_SegNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_DeepLab'):
        net = CMDNet_DeepLab(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_ConvUNeXt'):
        net = CMDNet_ConvUNeXt(in_channels = args.n_channels, num_classes = args.n_classes).cuda()
    else:
        print("the model name must be in CMDNet_FCN, CMDNet_FCNsa, CMDNet_UNet, CMDNet_ConvUNeXt.")


    net.eval()
    net = loadweight(net, args.model_path)
    test_ce_net_vessel(net,  args.test_path, args.gt_path)
    