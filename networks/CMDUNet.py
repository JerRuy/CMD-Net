
import torch
import torch.nn as nn
from networks.layers import unetConv2, unetUp
from thop import profile




class CMDUNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=False, is_batchnorm=True):
        super(CMDUNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        


        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)






        self.chx1 = nn.Conv2d(self.in_channels + filters[0], filters[0], 1, padding=0)  
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





        self.ch1 = nn.Conv2d(filters[0]  + filters[1], filters[0], 1, padding=0)  
        self.ch1b = nn.BatchNorm2d(filters[0])
        self.ch1r = nn.ReLU(inplace=True)
        
        self.ch2 = nn.Conv2d(filters[0] + filters[1] + filters[2], filters[1], 1, padding=0)
        self.ch2b = nn.BatchNorm2d(filters[1])
        self.ch2r = nn.ReLU(inplace=True)
        
        self.ch3 = nn.Conv2d(filters[1] + filters[2] + filters[3], filters[2], 1, padding=0)
        self.ch3b = nn.BatchNorm2d(filters[2])
        self.ch3r = nn.ReLU(inplace=True)
        
        self.ch4 = nn.Conv2d(filters[2] + filters[3] + filters[4], filters[3], 1, padding=0)
        self.ch4b = nn.BatchNorm2d(filters[3])
        self.ch4r = nn.ReLU(inplace=True)





        self.chy2 = nn.Conv2d(filters[2] + filters[1], filters[1], 1, padding=0)
        self.chy2b = nn.BatchNorm2d(filters[1])
        self.chy2r = nn.ReLU(inplace=True)
        
        self.chy3 = nn.Conv2d(filters[3] + filters[2], filters[2], 1, padding=0)
        self.chy3b = nn.BatchNorm2d(filters[2])
        self.chy3r = nn.ReLU(inplace=True)
        
        self.chy4 = nn.Conv2d(filters[4] + filters[3], filters[3], 1, padding=0)
        self.chy4b = nn.BatchNorm2d(filters[3])
        self.chy4r = nn.ReLU(inplace=True)

        self.chy5 = nn.Conv2d(filters[3] + filters[4], filters[4], 1, padding=0)  
        self.chy5b = nn.BatchNorm2d(filters[4])
        self.chy5r = nn.ReLU(inplace=True)
        

       
    def forward(self, inputs):
        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256
        
        
        #Conv2#############################        
        inputsd = self.maxpool(inputs)  
        x1 = torch.cat([inputsd,maxpool1],dim=1)
        x1 = self.chx1(x1)
        x1 = self.chx1b(x1)
        x1 = self.chx1r(x1)
        
        conv2 = self.conv2(x1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128


        #Conv3#############################
        maxpool1d = self.maxpool(maxpool1)  
        x2 = torch.cat([maxpool2,maxpool1d],dim=1)
        x2 = self.chx2(x2) 
        x2 = self.chx2b(x2)
        x2 = self.chx2r(x2)
         
        conv3 = self.conv3(x2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64


        #Conv4#############################
        maxpool2d = self.maxpool(maxpool2)  
        x3 = torch.cat([maxpool3,maxpool2d],dim=1)
        x3 = self.chx3(x3) 
        x3 = self.chx3b(x3)
        x3 = self.chx3r(x3)
        
        conv4 = self.conv4(x3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32


        #Conv5#############################
        maxpool3d = self.maxpool(maxpool3)  
        x4 = torch.cat([maxpool4,maxpool3d],dim=1)
        x4 = self.chx4(x4) 
        x4 = self.chx4b(x4)
        x4 = self.chx4r(x4)

        center = self.center(x4)       # 256*32*32
        #Conv4#############################





        
        #############################
        #maxpool4u = self.up(maxpool4)
        y5 = torch.cat([maxpool4,center],dim=1)
        y5 = self.chy5(y5)
        y5 = self.chy5b(y5)
        y5 = self.chy5r(y5)          
        
        centeru = self.up(center)
        ex4 = torch.cat([centeru,conv4,maxpool3],dim=1)
        ex4 = self.ch4(ex4)
        ex4 = self.ch4b(ex4)
        ex4 = self.ch4r(ex4)  
        
        up4 = self.up_concat4(y5,ex4)  # 128*64*64
  
        
        
        #############################
        centeru = self.up(center)
        y4 = torch.cat([centeru,up4],dim=1)
        y4 = self.chy4(y4)
        y4 = self.chy4b(y4)
        y4 = self.chy4r(y4)              

        conv4u = self.up(conv4)
        ex3 = torch.cat([conv4u,conv3,maxpool2],dim=1)
        ex3 = self.ch3(ex3)
        ex3 = self.ch3b(ex3)
        ex3 = self.ch3r(ex3)           
        up3 = self.up_concat3(y4,ex3)     # 64*128*128
        
        
        
        #############################
        up4u = self.up(up4)
        y3 = torch.cat([up4u,up3],dim=1)
        y3 = self.chy3(y3)
        y3 = self.chy3b(y3)
        y3 = self.chy3r(y3)
        
        conv3u = self.up(conv3)
        ex2 = torch.cat([conv3u,conv2,maxpool1],dim=1)
        ex2 = self.ch2(ex2)
        ex2 = self.ch2b(ex2)
        ex2 = self.ch2r(ex2)                    
        
        up2 = self.up_concat2(y3,ex2)     # 32*256*256
        
       
       
        #############################
        up3u = self.up(up3)
        y2 = torch.cat([up3u,up2],dim=1)
        y2 = self.chy2(y2)
        y2 = self.chy2b(y2)
        y2 = self.chy2r(y2)
        
        conv2u = self.up(conv2)
        ex1 = torch.cat([conv2u,conv1],dim=1)
        ex1 = self.ch1(ex1)
        ex1 = self.ch1b(ex1)
        ex1 = self.ch1r(ex1)          
        up1 = self.up_concat1(y2,ex1)     # 16*512*512


        final = self.final(up1)

        return final

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(4,3,256,256)).cuda()
    input = Variable(torch.rand(4,3,256,256)).cuda()

    model = CMDUNet().cuda()
    
    flops, params = profile(model, inputs=(input, ))
    y = model(x)

    print('Output shape:',y.shape)
    print('CMDUNet totoal parameters: %.2fM (%d)'%(params/1e6,params))
    print('CMDUNet totoal FLOPs     : %.2fG (%d)'%(flops/1e9,flops))
    #print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
