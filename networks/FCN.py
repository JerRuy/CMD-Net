#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class FCN(nn.Module): 
    def __init__(self, in_channels=3, n_classes=2, pretrained=False):
        super(FCN, self).__init__()
        
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.conv2_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.conv3_block = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.conv4_block = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.conv5_block = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, n_classes, 1),
        )
        
        
        #self.score_pool3 = nn.Conv2d(256, n_classes, 1)
        
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        #print(conv2.size())
        conv3 = self.conv3_block(conv2)
        #print(conv3.size())
        conv4 = self.conv4_block(conv3)
        #print(conv4.size())
        conv5 = self.conv5_block(conv4)
        #score = self.classifier(conv5)
        

        #print(conv5.size())
        score = self.relu(self.deconv1(conv5))               # size=(N, 512, x.H/16, x.W/16)
        #print(score.size(), conv4.size())
        score = self.bn1(score + conv4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        #print(score.size(), conv3.size())
        score = self.bn2(score + conv3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        #print(score.size())
        score = self.deconv3(score)  # size=(N, 128, x.H/4, x.W/4)
        #print(score.size())
        score = self.deconv4(score)  # size=(N, 64, x.H/2, x.W/2)
        #print(score.size())
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        #print(score.size())
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        #print(score.size())
        
        return torch.sigmoid(score), conv5  # size=(N, n_class, x.H/1, x.W/1)










