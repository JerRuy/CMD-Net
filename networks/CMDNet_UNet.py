# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

class CMDNet_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CMDNet_UNet, self).__init__()
        self.inc = inconv(n_channels, 64)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.encoderx1 = encoderx1(3,64)
        self.convx1 = double_conv(128,128)
        
        self.encoderx2 = encoderx2(3,64,128)
        self.convx2 = double_conv(256,256)
        
        self.encoderx3 = encoderx3(3,64,128,256)
        self.convx3 = double_conv(512,512)
        
        self.encoderx4 = encoderx4(3,64,128,256,512)
        self.convx4 = double_conv(1024,1024)
        
        self.ch = ch(1024,512)
        self.center = double_conv(512,512)
            
        self.encodery5 = encodery5(512,512)
        self.chy5 = ch(1024,256)
        self.convy5 = double_conv(256,256)
           
        self.encodery4 = encodery4(512,256,256)
        self.chy4 = ch(512,128)
        self.convy4 = double_conv(128,128)        
        
        self.encodery3 = encodery3(512,256,128,128)
        self.chy3 = ch(256,64)
        self.convy3 = double_conv(64,64)  
        
        self.encodery2 = encodery2(512,256,128,64,64)
        self.chy2 = ch(128,64)
        self.convy2 = double_conv(64,64)  
    
        
        self.encodery5x = encodery5x(512,512,256,128,64,512)
        self.chy5e = ch(1024,512)
        self.encodery4x = encodery4x(512,512,256,128,64,256)
        self.chy4e = ch(512,256)
        self.encodery3x = encodery3x(512,512,256,128,64,128)
        self.chy3e = ch(256,128)
        self.encodery2x = encodery2x(512,512,256,128,64,64)
        self.chy2e = ch(128,64)
        
        
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)
         
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        
        #x2 = self.down1(x1)
        #x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        
        
        ########################3,64,128,128
        x1d = self.down1(x1)        
        x1e = self.encoderx1(x)
        x1e=F.dropout(x1e, p=0.5, training=self.training)
        
        x1c = torch.cat([x1d,x1e],dim=1)
               
        x2 = self.convx1(x1c)
        ########################128,128,256,256
        x2d = self.down2(x2)
        x2e = self.encoderx2(x,x1c)
        x2e=F.dropout(x2e, p=0.5, training=self.training)
        
        x2c = torch.cat([x2d,x2e],dim=1)  
              
        x3 = self.convx2(x2c)
        ########################256,256,512,512
        x3d = self.down3(x3)
        x3e = self.encoderx3(x,x1c,x2c)
        x3e=F.dropout(x3e, p=0.5, training=self.training)
        
        x3c = torch.cat([x3d,x3e],dim=1)    
            
        x4 = self.convx3(x3c)        
    
        #x4=F.dropout(x4, p=0.5, training=self.training)
        ########################512,512,1024,1024,512
        x4d = self.down4(x4)
        x4e = self.encoderx4(x,x1c,x2c,x3c) 
        x4e=F.dropout(x4e, p=0.5, training=self.training)
               
        x4c = torch.cat([x4d,x4e],dim=1)   
             
        x5 = self.convx4(x4c)     
        #x5=F.dropout(x5, p=0.5, training=self.training)
        
        
        #########################################################
        #########################################################
        #########################################################
        
        
        x5 = self.ch(x5)
        y5 = self.center(x5)
        #y5 = x5
        
        
        #########################################################
        #########################################################
        #########################################################
              
        
        y5p = self.up1(y5)   
        
        y5ex = self.encodery5x(x5,x4,x3,x2,x1)
        y5ey = self.encodery5(x5)  
        
        y5e = torch.cat([y5ex,y5ey],dim=1) 
        y5e = self.chy5e(y5e)
        y5e=F.dropout(y5e, p=0.5, training=self.training)
        
        y5c = torch.cat([y5p,y5e],dim=1) 
        y5c = self.chy5(y5c)
        y4 = self.convy5(y5c)
    
        
        
        y4p = self.up2(y4)      
        
        y4ex = self.encodery4x(x5,x4,x3,x2,x1)
        y4ey = self.encodery4(x5,y5c)
        
        y4e = torch.cat([y4ex,y4ey],dim=1) 
        y4e = self.chy4e(y4e)        
        y4e=F.dropout(y4e, p=0.5, training=self.training)
        
        y4c = torch.cat([y4p,y4e],dim=1) 
        y4c = self.chy4(y4c)
        y3 = self.convy4(y4c)
        
        
        
        
        y3p = self.up3(y3)      
        
        y3ex = self.encodery3x(x5,x4,x3,x2,x1)
        y3ey = self.encodery3(x5,y5c,y4c)
        
        #print(y3ex.size(),y3ey.size())
        y3e = torch.cat([y3ex,y3ey],dim=1) 
        y3e = self.chy3e(y3e) 
        y3e=F.dropout(y3e, p=0.5, training=self.training)
        
        y3c = torch.cat([y3p,y3e],dim=1) 
        y3c = self.chy3(y3c)
        y2 = self.convy3(y3c)
        
        
        
        y2p = self.up4(y2)     
        
        y2ex = self.encodery2x(x5,x4,x3,x2,x1)
        y2ey = self.encodery2(x5,y5c,y4c,y3c)
                
        y2e = torch.cat([y2ex,y2ey],dim=1) 
        y2e = self.chy2e(y2e) 
        y2e=F.dropout(y2e, p=0.5, training=self.training)
        
        y2c = torch.cat([y2p,y2e],dim=1) 
        y2c = self.chy2(y2c)
        y1 = self.convy2(y2c)
        
        y = self.outc(y1)
        y = torch.sigmoid(y)
        return y, y5


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

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
            nn.Conv2d(in_ch, out_ch, 1)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
        
        
######################################################################################
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
        
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
        
        
        
        
        
######################################################################################
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x

class downn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downn, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x
        
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
            nn.Conv2d(in_ch2*2, int(out_ch/2), 1)
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
            nn.Conv2d(in_ch2*2, int(out_ch/4), 1),
            nn.MaxPool2d(4)
            
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_ch3*2, int(out_ch/2), 1),
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
            nn.Conv2d(in_ch1*2, int(out_ch/8), 1)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(in_ch2*2, int(out_ch/4), 1)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch3*2, int(out_ch/2), 1)
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
class encodery5x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3, in_ch4, in_ch5, out_ch):
        super(encodery5x, self).__init__()
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )  
        
        self.ch = nn.Conv2d(in_ch2,int(out_ch/2),1)
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/8), 1),
            nn.MaxPool2d(2)
            
        )      
        
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/16), 1),
            nn.MaxPool2d(4)
            
        )  
        
        self.down2 = nn.Sequential(
            nn.Conv2d(in_ch5, int(out_ch/16), 1),
            nn.MaxPool2d(8)
            
        )  
    def forward(self, x5,x4,x3,x2,x1):
        
        x5 = self.up(x5)
        x4 = self.ch(x4)
        x3 = self.down(x3)
        x2 = self.down1(x2)
        x1 = self.down2(x1)
        x = torch.cat([x1,x2,x3,x4,x5],dim=1)
        return x        
        
class encodery4x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,in_ch5,out_ch):
        super(encodery4x, self).__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/16), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )          
        
             
        self.up = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )      
                  
        self.ch = nn.Conv2d(in_ch3,int(out_ch/2),1)
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/8), 1),
            nn.MaxPool2d(2)
            
        )  
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch5, int(out_ch/16), 1),
            nn.MaxPool2d(4)
            
        )  
 
    def forward(self, x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x5 = self.up1(x5)
        x4 = self.up(x4)
        x3 = self.ch(x3)
        x2 = self.down(x2)
        x1 = self.down1(x1)
        
        x = torch.cat([x1,x2,x3,x4,x5],dim=1)
        return x            
 
 
class encodery3x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,in_ch5,out_ch):
        super(encodery3x, self).__init__()
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/16), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )      
        
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/16), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        ) 
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        ) 
                  
        self.ch = nn.Conv2d(in_ch4,int(out_ch/2),1)
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch5, int(out_ch/8), 1),
            nn.MaxPool2d(2)
            
        )  
    def forward(self, x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x5 = self.up2(x5)
        x4 = self.up1(x4)
        x3 = self.up(x3)
        x2 = self.ch(x2)
        x1 = self.down(x1)

        x = torch.cat([x5,x4,x3,x2,x1],dim=1)
        return x        
        
class encodery2x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3,in_ch4,in_ch5, out_ch):
        super(encodery2x, self).__init__()
        
        
        
        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/16), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        ) 
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/16), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        ) 
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/8), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        ) 
        
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )  
   
        self.ch = nn.Conv2d(in_ch5,int(out_ch/2),1)
    
    def forward(self, x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x5 = self.up3(x5)
        x4 = self.up2(x4)
        x3 = self.up1(x3)
        x2 = self.up(x2)
        x1 = self.ch(x1)

        x = torch.cat([x5,x4,x3,x2,x1],dim=1)
        return x           
        
        
        
        
###########################################################################
class encodery5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encodery5, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = single_conv(in_ch, out_ch)

    def forward(self, x5):
        x = self.up(x5)
        #x = self.conv(x)
        return x 

class encodery4(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(encodery4, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/2), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/2), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x5, y5c):
        x = self.up1(x5)
        y5c = self.up2(y5c)
        
        x = torch.cat([x,y5c],dim=1)
        #x = self.conv(x)
        return x 
        
        
        
class encodery3(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super(encodery3, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/4), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
           
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/2), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x, y5c,y4c):
        x = self.up1(x)
        y5c = self.up2(y5c)
        y4c = self.up3(y4c)
        
        x = torch.cat([x,y4c,y5c],dim=1)
        #x = self.conv(x)
        return x  
        
class encodery2(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4, out_ch):
        super(encodery2, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/8), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/8), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/4), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    
        self.up4 = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/2), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x, y5c,y4c,y3c):
        x = self.up1(x)
        y5c = self.up2(y5c)
        y4c = self.up3(y4c)
        y3c = self.up4(y3c)
        x = torch.cat([x,y5c,y4c,y3c],dim=1)
        #x = self.conv(x)
        return x      


class encodery1(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,in_ch5, out_ch):
        super(encodery1, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/16), 1),
            nn.Upsample(scale_factor=32, mode='bilinear')
            
        )
        
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/16), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/8), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/4), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    
        self.up5 = nn.Sequential(
            nn.Conv2d(in_ch5, int(out_ch/2), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x, y5c,y4c,y3c,y2c):
        x = self.up1(x)
        y5c = self.up2(y5c)
        y4c = self.up3(y4c)
        y3c = self.up4(y3c)
        x = torch.cat([x,y5c,y4c,y3c,y2c],dim=1)
        #x = self.conv(x)
        return x  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ######################################################################################
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(int(in_ch/2), int(in_ch/2), 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #print(x.size()[2] , x.size()[3] )
        return x


class upp1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upp1, self).__init__()
 
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(int(in_ch/2), int(in_ch/4), 1, padding=0)
        )
        #self.down = nn.Conv2d(int(in_ch/4), int(in_ch/4), 3, padding=2, stride=2, dilation=2)
        self.down = nn.MaxPool2d(2)
         
        self.conv = double_conv(in_ch, out_ch)        
        

    def forward(self, x1, x2, x3):#?��??��???
        x1 = self.up(x1)
        x3 = self.down(x3) 
  
                               
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv(x)
#        print(x.size()[2] , x.size()[3])
        return x
        
class upp2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upp2, self).__init__()
 
        self.upy = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(int(in_ch/2), int(in_ch/4), 1, padding=0)
        )
        self.upx = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, int(in_ch/4), 1, padding=0)
        )
         
        self.conv = double_conv(in_ch, out_ch)        
        

    def forward(self, y, x2, x1):#?��??��???
        y = self.upy(y)
        x2 = self.upx(x2) 
  
        #print(y.size(),x1.size(),x2.size())
        x = torch.cat([y, x1, x2], dim=1)
        x = self.conv(x)
#        print(x.size()[2] , x.size()[3])
        return x
      
class uppp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(uppp, self).__init__()
        ######### encoder #####################################################
        self.eup = nn.Upsample(scale_factor=2, mode='bilinear')  #in_ch/2 ???? in_ch/2
        self.edown = nn.Conv2d(int(in_ch/4), int(in_ch/2), 3, padding=2, stride=2, dilation=2) #in_ch/4 ???? in_ch/2
        
        self.encoder = nn.Conv2d(int(in_ch*2), int(in_ch/2), 1)
        
        ######### decoder #####################################################             
        self.dup = nn.Upsample(scale_factor=2, mode='bilinear')
        
        ######### finally #####################################################              
        self.conv = double_conv(in_ch, out_ch)   

    def forward(self, y, x1, x2, x3):#decoder:y encoder:?��??��???
    
        x1 = self.eup(x1)          #in_ch ???? in_ch
        x2 = x2                    #in_ch/2 ???? in_ch/2
        x3 = self.edown(x3)        #in_ch/4 ???? in_ch/2
        
        #print(x1.size()[0] ,x1.size()[1] ,x1.size()[2] , x1.size()[3] )   
        #print(x2.size()[0] ,x2.size()[1] ,x2.size()[2] , x2.size()[3] ) 
        #print(x3.size()[0] ,x3.size()[1] ,x3.size()[2] , x3.size()[3] ) 
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.encoder(x)        #in_ch*2 ???? in_ch/2
        
        y = self.dup(y)            #in_ch/2 ???? in_ch/2
#        print(x1.size()[2] , x1.size()[3]    ,x2.size()[2] , x2.size()[3]   ,x3.size()[2] , x3.size()[3] )   
                               
        z = torch.cat([x, y], dim=1)
        z = self.conv(z)           #in_ch ???? out_ch
#        print(x.size()[2] , x.size()[3] )
        return z