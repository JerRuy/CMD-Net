# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F



class CMDNet_FCN(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, pretrained=False):
        super(CMDNet_FCN, self).__init__()
        
        self.conv1_block = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
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


        filters = [64, 128, 256, 512, 512]
        self.chx1 = nn.Conv2d(n_channels + filters[0], filters[0], 1, padding=0)  
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
        
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)






        self.ch3 = nn.Conv2d(filters[1] + filters[2] + filters[3], filters[2], 1, padding=0)
        self.ch3b = nn.BatchNorm2d(filters[2])
        self.ch3r = nn.ReLU(inplace=True)
        
        self.ch4 = nn.Conv2d(filters[2] + filters[3] + filters[4], filters[3], 1, padding=0)
        self.ch4b = nn.BatchNorm2d(filters[3])
        self.ch4r = nn.ReLU(inplace=True)
        
        self.chy4 = nn.Conv2d(filters[4] + filters[3], filters[3], 1, padding=0)
        self.chy4b = nn.BatchNorm2d(filters[3])
        self.chy4r = nn.ReLU(inplace=True)

        self.chy5 = nn.Conv2d(filters[3] + filters[4], filters[4], 1, padding=0)  
        self.chy5b = nn.BatchNorm2d(filters[4])
        self.chy5r = nn.ReLU(inplace=True)




    def forward(self, x):
        x1 = self.conv1_block(x)
                
        xd = self.maxpool(x)  
        x1 = torch.cat([x1,xd],dim=1)
        x1 = self.chx1(x1)
        x1 = self.chx1b(x1)
        x1 = self.chx1r(x1)
        
        
        x2 = self.conv2_block(x1)

        x1d = self.maxpool(x1)  
        x2 = torch.cat([x2,x1d],dim=1)
        x2 = self.chx2(x2) 
        x2 = self.chx2b(x2)
        x2 = self.chx2r(x2)        
        #print(conv2.size())
        x3 = self.conv3_block(x2)

        x2d = self.maxpool(x2)  
        x3 = torch.cat([x3,x2d],dim=1)
        x3 = self.chx3(x3) 
        x3 = self.chx3b(x3)
        x3 = self.chx3r(x3)        
        #print(conv3.size())
        x4 = self.conv4_block(x3)
        
        x3d = self.maxpool(x3)  
        x4 = torch.cat([x4,x3d],dim=1)
        x4 = self.chx4(x4) 
        x4 = self.chx4b(x4)
        x4 = self.chx4r(x4)        
        #print(conv4.size())
        x5 = self.conv5_block(x4)
        

        y4 = self.deconv1(x5)
        #print(conv5.size())
        y4 = self.relu(y4)               # size=(N, 512, x.H/16, x.W/16)     
        #print(score.size(), conv4.size())
        y4 = self.bn1(y4 + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        

        
        #############################
        x5u = self.up(x5)
        y4 = torch.cat([x5u,y4],dim=1)
        y4 = self.chy4(y4)
        y4 = self.chy4b(y4)
        y4 = self.chy4r(y4)              

        x4u = self.up(x4)
        ex3 = torch.cat([x4u,x3,x2d],dim=1)
        ex3 = self.ch3(ex3)
        ex3 = self.ch3b(ex3)
        ex3 = self.ch3r(ex3)           
  
        
        #############################  
        
                
        
        y3 = self.deconv2(y4)
        y3 = self.relu(y3)    # size=(N, 256, x.H/8, x.W/8)
        #print(score.size(), conv3.size())
        y3 = self.bn2(y3 + ex3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        
        
        
        #print(score.size())
        y3 = self.deconv3(y3)  # size=(N, 128, x.H/4, x.W/4)
        #print(score.size())
        y2 = self.deconv4(y3)  # size=(N, 64, x.H/2, x.W/2)
        #print(score.size())
        score = self.bn5(self.relu(self.deconv5(y2)))  # size=(N, 32, x.H, x.W)
        #print(score.size())
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        #print(score.size())
        score = torch.sigmoid(score)
        return score, x5  # size=(N, n_class, x.H/1, x.W/1)










