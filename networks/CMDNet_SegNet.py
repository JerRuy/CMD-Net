#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

  
  
class CMDNet_SegNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, pretrained=False):
        super(CMDNet_SegNet, self).__init__()
        self.downn1 = segnetDown2(in_channels, 64)
        self.downn2 = segnetDown2(64, 128)
        self.downn3 = segnetDown3(128, 256)
        self.downn4 = segnetDown3(256, 512)
        self.downn5 = segnetDown3(512, 512)

        self.upp5 = segnetUp3(512, 512)
        self.upp4 = segnetUp3(512, 256)
        self.upp3 = segnetUp3(256, 128)
        self.upp2 = segnetUp2(128, 64)
        self.upp1 = segnetUp2(64, n_classes)

        #self.init_weights(pretrained=pretrained)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        filters = [64,128,256,512,512]

        self.chx1 = nn.Conv2d(3 + filters[0], filters[0], 1, padding=0)  
        self.chx1b = nn.BatchNorm2d(filters[0])
        self.chx1r = nn.ReLU(inplace=True)
        
        self.chx2 = nn.Conv2d(filters[0] + filters[1], filters[1], 1, padding=0)
        self.chx2b = nn.BatchNorm2d(filters[1])
        self.chx2r = nn.ReLU(inplace=True)
        
        self.chx3 = nn.Conv2d(filters[1] + filters[2], filters[2], 1, padding=0)
        self.chx3b = nn.BatchNorm2d(filters[2])
        self.chx3r = nn.ReLU(inplace=True)
        
        self.chx4 = nn.Conv2d(filters[2] + filters[3], filters[3], 1, padding=0)
        self.chx4b = nn.BatchNorm2d(filters[3])
        self.chx4r = nn.ReLU(inplace=True)


        self.chex1 = nn.Conv2d(filters[0]  + filters[1], filters[0], 1, padding=0)  
        self.chex1b = nn.BatchNorm2d(filters[0])
        self.chex1r = nn.ReLU(inplace=True)
        
        self.chex2 = nn.Conv2d(filters[0] + filters[1] + filters[2], filters[1], 1, padding=0)
        self.chex2b = nn.BatchNorm2d(filters[1])
        self.chex2r = nn.ReLU(inplace=True)
        
        self.chex3 = nn.Conv2d(filters[1] + filters[2] + filters[3], filters[2], 1, padding=0)
        self.chex3b = nn.BatchNorm2d(filters[2])
        self.chex3r = nn.ReLU(inplace=True)
        
        self.chex4 = nn.Conv2d(filters[2] + filters[3] + filters[4], filters[3], 1, padding=0)
        self.chex4b = nn.BatchNorm2d(filters[3])
        self.chex4r = nn.ReLU(inplace=True)



        self.chey1 = nn.Conv2d(filters[1] + filters[0], filters[0], 1, padding=0)
        self.chey1b = nn.BatchNorm2d(filters[0])
        self.chey1r = nn.ReLU(inplace=True)


        self.chey2 = nn.Conv2d(filters[2] + filters[1], filters[1], 1, padding=0)
        self.chey2b = nn.BatchNorm2d(filters[1])
        self.chey2r = nn.ReLU(inplace=True)
        
        self.chey3 = nn.Conv2d(filters[3] + filters[2], filters[2], 1, padding=0)
        self.chey3b = nn.BatchNorm2d(filters[2])
        self.chey3r = nn.ReLU(inplace=True)
        
        self.chey4 = nn.Conv2d(filters[4] + filters[3], filters[3], 1, padding=0)
        self.chey4b = nn.BatchNorm2d(filters[3])
        self.chey4r = nn.ReLU(inplace=True)

        self.chey5 = nn.Conv2d(filters[3] + filters[4], filters[4], 1, padding=0)  
        self.chey5b = nn.BatchNorm2d(filters[4])
        self.chey5r = nn.ReLU(inplace=True)



        self.ch1 = nn.Conv2d(filters[0]  + filters[0], filters[0], 1, padding=0)  
        self.ch1b = nn.BatchNorm2d(filters[0])
        self.ch1r = nn.ReLU(inplace=True)
        
        self.ch2 = nn.Conv2d(filters[1] + filters[1], filters[1], 1, padding=0)
        self.ch2b = nn.BatchNorm2d(filters[1])
        self.ch2r = nn.ReLU(inplace=True)
        
        self.ch3 = nn.Conv2d(filters[2] + filters[2], filters[2], 1, padding=0)
        self.ch3b = nn.BatchNorm2d(filters[2])
        self.ch3r = nn.ReLU(inplace=True)
        
        self.ch4 = nn.Conv2d( filters[3] + filters[3], filters[3], 1, padding=0)
        self.ch4b = nn.BatchNorm2d(filters[3])
        self.ch4r = nn.ReLU(inplace=True)


        self.down5 = nn.MaxPool2d(2)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down4 = nn.MaxPool2d(2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down3 = nn.MaxPool2d(2)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down2 = nn.MaxPool2d(2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.up44 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up33 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up22 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
    
        x1, pool_indices1, unpool_shape1 = self.downn1(x)
        
        
        maxpool1 = self.maxpool1(x) 
        x1 = torch.cat([x1,maxpool1],dim=1)
        x1 = self.chx1(x1)
        x1 = self.chx1b(x1)
        x1 = self.chx1r(x1)                        
        x2, pool_indices2, unpool_shape2 = self.downn2(x1)
        
        maxpool2 = self.maxpool2(x1)  
        x2 = torch.cat([x2,maxpool2],dim=1)
        x2 = self.chx2(x2) 
        x2 = self.chx2b(x2)
        x2 = self.chx2r(x2)
        x3, pool_indices3, unpool_shape3 = self.downn3(x2)
        
        maxpool3 = self.maxpool3(x2)  
        x3 = torch.cat([x3,maxpool3],dim=1)
        x3 = self.chx3(x3) 
        x3 = self.chx3b(x3)
        x3 = self.chx3r(x3)
        x4, pool_indices4, unpool_shape4 = self.downn4(x3)
        
        
        maxpool4 = self.maxpool4(x3)  
        x4 = torch.cat([x4,maxpool4],dim=1)
        x4 = self.chx4(x4) 
        x4 = self.chx4b(x4)
        x4 = self.chx4r(x4)         
        x5, pool_indices5, unpool_shape5 = self.downn5(x4)

        y5 = self.upp5(x5, pool_indices=pool_indices5, unpool_shape=unpool_shape5)        
        
        x5u = self.up5(x5)
        x3d = self.down5(x3)

        ex4 = torch.cat([x5u,x4,x3d],dim=1)
        ex4 = self.chex4(ex4)
        ex4 = self.chex4b(ex4)
        ex4 = self.chex4r(ex4)          
        
        y5 = torch.cat([ex4,y5],dim=1)
        y5 = self.ch4(y5)
        y5 = self.ch4b(y5)
        y5 = self.ch4r(y5)       
                
        y4 = self.upp4(y5, pool_indices=pool_indices4, unpool_shape=unpool_shape4)
        
        x4u = self.up4(x4)
        x2d = self.down4(x2)
        
        ex3 = torch.cat([x4u,x3,x2d],dim=1)
        ex3 = self.chex3(ex3)   
        ex3 = self.chex3b(ex3)
        ex3 = self.chex3r(ex3) 
                
        y5u = self.up44(y5)
        ey3 = torch.cat([y4,y5u],dim=1)
        ey3 = self.chey3(ey3)
        ey3 = self.chey3b(ey3)
        ey3 = self.chey3r(ey3) 
               
               
        y4 = torch.cat([ex3,ey3],dim=1)
        y4 = self.ch3(y4)
        y4 = self.ch3b(y4)
        y4 = self.ch3r(y4)  

        
        y3 = self.upp3(y4, pool_indices=pool_indices3, unpool_shape=unpool_shape3)


        x3u = self.up3(x3)
        x1d = self.down3(x1)
        
        ex2 = torch.cat([x3u,x2,x1d],dim=1)
        ex2 = self.chex2(ex2)  
        ex2 = self.chex2b(ex2)
        ex2 = self.chex2r(ex2)  
        
        y4u = self.up33(y4)
        ey2 = torch.cat([y3,y4u],dim=1)
        ey2 = self.chey2(ey2)
        ey2 = self.chey2b(ey2)
        ey2 = self.chey2r(ey2)        
        
        y3 = torch.cat([ex2,ey2],dim=1)
        y3 = self.ch2(y3)
        y3 = self.ch2b(y3)
        y3 = self.ch2r(y3) 
        
        y2 = self.upp2(y3, pool_indices=pool_indices2, unpool_shape=unpool_shape2)
        

        x2u = self.up2(x2)
        ex1 = torch.cat([x1,x2u],dim=1)
        ex1 = self.chex1(ex1) 
        ex1 = self.chex1b(ex1)
        ex1 = self.chex1r(ex1)   
        
        y3u = self.up22(y3)
        ey1 = torch.cat([y2,y3u],dim=1)
        ey1 = self.chey1(ey1)
        ey1 = self.chey1b(ey1)
        ey1 = self.chey1r(ey1) 
        
        y2 = torch.cat([ex1,ey1],dim=1)
        y2 = self.ch1(y2)
        y2 = self.ch1b(y2)
        y2 = self.ch1r(y2) 


        y1 = self.upp1(y2, pool_indices=pool_indices1, unpool_shape=unpool_shape1)
        y1 = torch.sigmoid(y1)
        return y1, x5



class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()
        self.cbr_seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cbr_seq(x)
        return x
        
class segnetDown2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpool_shape = x.size()
        # print(unpool_shape)
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape


class segnetDown3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpool_shape = x.size()
        # print(unpool_shape)
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape

class segnetDown4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown4, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        unpool_shape = x.size()
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape


class segnetUp2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp2, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class segnetUNetDown2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUNetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpool_shape = x.size()
        x_pool, pool_indices = self.max_pool(x)
        return x_pool, pool_indices, unpool_shape, x


class segnetUNetDown3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUNetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpool_shape = x.size()
        x_pool, pool_indices = self.max_pool(x)
        return x_pool, pool_indices, unpool_shape, x

class segnetUp3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp3, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class segnetUp4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp4, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x







