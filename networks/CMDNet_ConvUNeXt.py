from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


class Conv(nn.Module):
    def __init__(self, dim):
        super(Conv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        x = self.act2(residual + x)

        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(Conv(out_channels))
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_num=1):
        super(Up, self).__init__()
        C = in_channels // 2
        self.norm = nn.BatchNorm2d(C)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.gate = nn.Linear(C, 3 * C)
        self.linear1 = nn.Linear(C, C)
        self.linear2 = nn.Linear(C, C)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(Conv(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x1)
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        #attention
        B, C, H, W = x1.shape
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        gate = self.gate(x1).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        g1, g2, g3 = gate[0], gate[1], gate[2]
        x2 = torch.sigmoid(self.linear1(g1 + x2)) * x2 + torch.sigmoid(g2) * torch.tanh(g3)
        x2 = self.linear2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class CMDNet_ConvUNeXt(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(CMDNet_ConvUNeXt, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.GELU(),
            Conv(base_c)
        )

        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8, layer_num=3)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)


        self.encoderx1 = encoderx1(3,32)
        self.encoderx2 = encoderx2(3,64,128)
        self.encoderx3 = encoderx3(3,64,128,256)
        self.encoderx4 = encoderx4(3,64,128,256,1024)

        
        self.chx1 = ch(96, 64)
        self.chx2 = ch(256,128)
        self.chx3 = ch(512,256)
        self.chx4 = ch(1280,256)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        ########################
        x2 = self.down1(x1)
        x1e = self.encoderx1(x)
        x2 = torch.cat([x2, x1e],dim=1)
        x2 = self.chx1(x2)
        ########################
        
        x3 = self.down2(x2)
        x2e = self.encoderx2(x, x2)
        x3 = torch.cat([x3, x2e],dim=1)
        x3 = self.chx2(x3)
        ########################

        x4 = self.down3(x3)
        x3e = self.encoderx3(x, x2, x3)
        x4 = torch.cat([x4,x3e],dim=1)
        x4 = self.chx3(x4)
        ########################

        x5 = self.down4(x4)
        x4e = self.encoderx4(x, x2, x3, x4)
        x5 = torch.cat([x5,x4e],dim=1)
        x5 = self.chx4(x5)
        ########################

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        logits = torch.sigmoid(logits)
        return logits, x5


####encoderx##########################################################################
        
class encoderx1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encoderx1, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = single_conv(in_ch, out_ch)
        self.ch = nn.Conv2d(in_ch, int(out_ch), 1)
    def forward(self, x):
        x = self.down(x)
        #x = self.conv(x)
        x = self.ch(x)
        return x        

class encoderx2(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(encoderx2, self).__init__()
        self.down1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(in_ch1, int(out_ch/2), 1)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch2, int(out_ch/2), 1)
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x, x1c):
        x = self.down1(x)
        x1c = self.down2(x1c)
        
        x = torch.cat([x,x1c],dim=1)
       # x = self.conv(x)
        return x         


class encoderx3(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(encoderx3, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/4), 1),
            nn.MaxPool2d(8)
            
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.MaxPool2d(4)
            
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/2), 1),
            nn.MaxPool2d(2)
            
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x, x1c,x2c):
        x = self.down1(x)
        x1c = self.down2(x1c)
        x2c = self.down3(x2c)
        
        x = torch.cat([x,x1c,x2c],dim=1)
       # x = self.conv(x)
        return x     

class encoderx4(nn.Module):
    def __init__(self, in_ch0, in_ch1, in_ch2, in_ch3, out_ch):
        super(encoderx4, self).__init__()
        self.down0 = nn.Sequential(
            nn.MaxPool2d(16),
            nn.Conv2d(in_ch0, int(out_ch/8), 1)
        )
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(8),
            nn.Conv2d(in_ch1, int(out_ch/8), 1)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(in_ch2, int(out_ch/4), 1)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch3, int(out_ch/2), 1)
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x0,x, x1c,x2c):
        x0 = self.down0(x0)
        x = self.down1(x)
        x1c = self.down2(x1c)
        x2c = self.down3(x2c)
        
        x = torch.cat([x0,x,x1c,x2c],dim=1)
        #x = self.conv(x)
        return x      
        
     
###########################################################################
###########################################################################    
###########################################################################

class single_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ch, self).__init__()         
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x