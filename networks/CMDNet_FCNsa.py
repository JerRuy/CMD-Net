# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMDNet_FCNsa(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CMDNet_FCNsa, self).__init__()

        self.down = nn.MaxPool2d(2)



        self.inc = inconv(n_channels, 32)
        
        self.encoderx1 = encoderx1(3,32)
        self.chx1 = ch(64,32)
        self.convx1 = bottle_conv1(256,256)
        
        self.encoderx2 = encoderx2(3,32,256)
        self.chx2 = ch(512,256)
        self.convx2 = bottle_conv2(512,512)
        
        self.encoderx3 = encoderx3(3,32,256,512)
        self.chx3 = ch(1024,512)
        self.convx3 = bottle_conv2(1024,1024)
        
        self.encoderx4 = encoderx4(3,32,256,512,1024)
        self.chx4 = ch(2048,1024)
        self.convx4 = bottle_conv3(1024,1024)
        
        self.encoderx5 = encoderx5(3,32,256,512,1024,1024)
        self.chx5 = ch(2048,1024)
        self.convx5 = bottle_conv3(1024,1024)



################################################################################################        
################################################################################################        
################################################################################################        
        ########################
        
        self.n6upy6 = upd(1024,6 * n_classes)
        self.n6encodery6x = n6encodery6x(1024,1024,1024,512,256,32,   32 * n_classes)    
        self.n6chy6e = ch(32 * n_classes ,                  6 * n_classes)        

        self.n6chy6 = ch(6 * n_classes * 2 ,6 * n_classes)
        self.n6convy6 = double_conv(6 * n_classes  ,6 * n_classes)       
        ########################        
        
        
        
        self.n6upy5 = upd(6 * n_classes,5 * n_classes)
        self.n6encodery5x = n6encodery5x(1024,1024,1024,512,256,32,   32 * n_classes)    
        self.n6encodery5 = n6encodery5(1024 ,          32 * n_classes)
        self.n6chy5e = ch(32 * n_classes *2 ,                  5 * n_classes)        

        self.n6chy5 = ch(5 * n_classes * 2 ,5 * n_classes)
        self.n6convy5 = double_conv(5 * n_classes  ,5 * n_classes)       
        ########################
        

        
        self.n6upy4 = upd(5 * n_classes,4 * n_classes)
        self.n6encodery4x = n6encodery4x(1024,1024,1024,512,256,32,    32 * n_classes)
        self.n6encodery4 = n6encodery4(1024 , 6 * n_classes,           32 * n_classes)
                       
        self.n6chy4e = ch(32 * n_classes * 2 ,4 * n_classes)
        self.n6chy4 = ch(4 * n_classes * 2 ,4 * n_classes)
        self.n6convy4 = double_conv(4 * n_classes ,4 * n_classes)          
        ########################
        

        self.n6upy3 = upd(4 * n_classes,3 * n_classes)
        self.n6encodery3x = n6encodery3x(1024,1024,1024,512,256,32,    32 * n_classes)
        self.n6encodery3 = n6encodery3(1024,6 * n_classes,5 * n_classes ,       32 * n_classes)
                       
        self.n6chy3e = ch(32 * n_classes * 2 ,3 * n_classes)
        self.n6chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.n6convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.n6upy2 = upd(3 * n_classes,2 * n_classes)
        self.n6encodery2x = n6encodery2x(1024,1024,1024,512,256,32,   32 * n_classes)
        self.n6encodery2 = n6encodery2(1024,6 * n_classes,5 * n_classes,4 * n_classes ,            32 * n_classes)
                       
        self.n6chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.n6chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.n6convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.n6out = outconv(2 * n_classes, n_classes)

        

        

################################################################################################        
################################################################################################        
################################################################################################        
        ########################
        self.n5upy5 = upd(1024,5 * n_classes)
        self.n5encodery5x = n5encodery5x(1024,1024,512,256,32,   32 * n_classes)    
        self.n5chy5e = ch(32 * n_classes ,                  5 * n_classes)        

        self.n5chy5 = ch(5 * n_classes * 2 ,5 * n_classes)
        self.n5convy5 = double_conv(5 * n_classes  ,5 * n_classes)       
        ########################
        

        
        self.n5upy4 = upd(5 * n_classes,4 * n_classes)
        self.n5encodery4x = n5encodery4x(1024,1024,512,256,32,    32 * n_classes)
        self.n5encodery4 = n5encodery4(1024 ,                     32 * n_classes)
                       
        self.n5chy4e = ch(32 * n_classes * 2 ,4 * n_classes)
        self.n5chy4 = ch(4 * n_classes * 2 ,4 * n_classes)
        self.n5convy4 = double_conv(4 * n_classes ,4 * n_classes)          
        ########################
        

        self.n5upy3 = upd(4 * n_classes,3 * n_classes)
        self.n5encodery3x = n5encodery3x(1024,1024,512,256,32,    32 * n_classes)
        self.n5encodery3 = n5encodery3(1024,5 * n_classes ,       32 * n_classes)
                       
        self.n5chy3e = ch(32 * n_classes * 2 ,3 * n_classes)
        self.n5chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.n5convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.n5upy2 = upd(3 * n_classes,2 * n_classes)
        self.n5encodery2x = n5encodery2x(1024,1024,512,256,32,   32 * n_classes)
        self.n5encodery2 = n5encodery2(1024,5 * n_classes,4 * n_classes ,            32 * n_classes)
                       
        self.n5chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.n5chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.n5convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.n5out = outconv(2 * n_classes, n_classes)
        
################################################################################################
################################################################################################
################################################################################################

        
        
        self.n4upy4 = upd(1024,4 * n_classes)
        self.n4encodery4x = n4encodery4x(1024,512,256,32,    32 * n_classes)
        #self.n4encodery4 = n4encodery4(1024 ,                     32 * n_classes)
                       
        self.n4chy4e = ch(32 * n_classes  ,4 * n_classes)
        self.n4chy4 = ch(4 * n_classes * 2 ,4 * n_classes)
        self.n4convy4 = double_conv(4 * n_classes ,4 * n_classes)          
        ########################
        

        self.n4upy3 = upd(4 * n_classes,3 * n_classes)
        self.n4encodery3x = n4encodery3x(1024,512,256,32,    32 * n_classes)
        self.n4encodery3 = n4encodery3(1024,       32 * n_classes)
                       
        self.n4chy3e = ch(32 * n_classes * 2 ,3 * n_classes)
        self.n4chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.n4convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.n4upy2 = upd(3 * n_classes,2 * n_classes)
        self.n4encodery2x = n4encodery2x(1024,512,256,32,   32 * n_classes)
        self.n4encodery2 = n4encodery2(1024,4 * n_classes ,            32 * n_classes)
                       
        self.n4chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.n4chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.n4convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.n4out = outconv(2 * n_classes, n_classes)

################################################################################################
################################################################################################
################################################################################################


        

        self.n3upy3 = upd(512,3 * n_classes)
        self.n3encodery3x = n3encodery3x(512,256,32,    32 * n_classes)
        #self.n3encodery3 = n3encodery3(512,256,       32 * n_classes)
                       
        self.n3chy3e = ch(32 * n_classes  ,3 * n_classes)
        self.n3chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.n3convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.n3upy2 = upd(3 * n_classes,2 * n_classes)
        self.n3encodery2x = n3encodery2x(512,256,32,   32 * n_classes)
        self.n3encodery2 = n3encodery2(512,            32 * n_classes)
                       
        self.n3chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.n3chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.n3convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.n3out = outconv(2 * n_classes, n_classes)
################################################################################################
################################################################################################
################################################################################################



        self.n2upy2 = upd(256,2 * n_classes)
        self.n2encodery2x = n2encodery2x(256,32,   32 * n_classes)
        #self.n2encodery2 = n2encodery2(512,            32 * n_classes)
                       
        self.n2chy2e = ch(32 * n_classes  ,2 * n_classes)
        self.n2chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.n2convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.n2out = outconv(2 * n_classes, n_classes)
################################################################################################
################################################################################################
################################################################################################


        self.n1convy1 = double_conv(32 ,2 * n_classes)  
        self.n1out = outconv(2 * n_classes, n_classes)

################################################################################################
################################################################################################
################################################################################################


        self.nout = outconv(6 * n_classes, n_classes)











 ################################################################################################        
################################################################################################        
################################################################################################       
################################################################################################        
################################################################################################        
################################################################################################
################################################################################################        
################################################################################################        
################################################################################################       







 
        ########################
        
        self.nn6upy6 = upd(1024,6 * n_classes)
        self.nn6encodery6x = n6encodery6x(1024,1024,1024,512,256,32,   32 * n_classes)    
        self.nn6chy6e = ch(32 * n_classes ,                  6 * n_classes)        

        self.nn6chy6 = ch(6 * n_classes * 2 ,6 * n_classes)
        self.nn6convy6 = double_conv(6 * n_classes  ,6 * n_classes)       
        ########################        
        
        
        
        self.nn6upy5 = upd(6 * n_classes,5 * n_classes)
        self.nn6encodery5x = n6encodery5x(1024,1024,1024,512,256,32,   32 * n_classes)    
        self.nn6encodery5 = n6encodery5(1024 ,          32 * n_classes)
        self.nn6chy5e = ch(32 * n_classes *2 ,                  5 * n_classes)        

        self.nn6chy5 = ch(5 * n_classes * 2 ,5 * n_classes)
        self.nn6convy5 = double_conv(5 * n_classes  ,5 * n_classes)       
        ########################
        

        
        self.nn6upy4 = upd(5 * n_classes,4 * n_classes)
        self.nn6encodery4x = n6encodery4x(1024,1024,1024,512,256,32,    32 * n_classes)
        self.nn6encodery4 = n6encodery4(1024 , 6 * n_classes,           32 * n_classes)
                       
        self.nn6chy4e = ch(32 * n_classes * 2 ,4 * n_classes)
        self.nn6chy4 = ch(4 * n_classes * 2 ,4 * n_classes)
        self.nn6convy4 = double_conv(4 * n_classes ,4 * n_classes)          
        ########################
        

        self.nn6upy3 = upd(4 * n_classes,3 * n_classes)
        self.nn6encodery3x = n6encodery3x(1024,1024,1024,512,256,32,    32 * n_classes)
        self.nn6encodery3 = n6encodery3(1024,6 * n_classes,5 * n_classes ,       32 * n_classes)
                       
        self.nn6chy3e = ch(32 * n_classes * 2 ,3 * n_classes)
        self.nn6chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.nn6convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.nn6upy2 = upd(3 * n_classes,2 * n_classes)
        self.nn6encodery2x = n6encodery2x(1024,1024,1024,512,256,32,   32 * n_classes)
        self.nn6encodery2 = n6encodery2(1024,6 * n_classes,5 * n_classes,4 * n_classes ,            32 * n_classes)
                       
        self.nn6chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.nn6chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.nn6convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.nn6out = outconv(2 * n_classes, n_classes)

        

################################################################################################        
################################################################################################        
################################################################################################  

      
        ########################
        self.nn5upy5 = upd(1024,5 * n_classes)
        self.nn5encodery5x = n5encodery5x(1024,1024,512,256,32,   32 * n_classes)    
        self.nn5chy5e = ch(32 * n_classes ,                  5 * n_classes)        

        self.nn5chy5 = ch(5 * n_classes * 2 ,5 * n_classes)
        self.nn5convy5 = double_conv(5 * n_classes  ,5 * n_classes)       
        ########################
        

        
        self.nn5upy4 = upd(5 * n_classes,4 * n_classes)
        self.nn5encodery4x = n5encodery4x(1024,1024,512,256,32,    32 * n_classes)
        self.nn5encodery4 = n5encodery4(1024 ,                     32 * n_classes)
                       
        self.nn5chy4e = ch(32 * n_classes * 2 ,4 * n_classes)
        self.nn5chy4 = ch(4 * n_classes * 2 ,4 * n_classes)
        self.nn5convy4 = double_conv(4 * n_classes ,4 * n_classes)          
        ########################
        

        self.nn5upy3 = upd(4 * n_classes,3 * n_classes)
        self.nn5encodery3x = n5encodery3x(1024,1024,512,256,32,    32 * n_classes)
        self.nn5encodery3 = n5encodery3(1024,5 * n_classes ,       32 * n_classes)
                       
        self.nn5chy3e = ch(32 * n_classes * 2 ,3 * n_classes)
        self.nn5chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.nn5convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.nn5upy2 = upd(3 * n_classes,2 * n_classes)
        self.nn5encodery2x = n5encodery2x(1024,1024,512,256,32,   32 * n_classes)
        self.nn5encodery2 = n5encodery2(1024,5 * n_classes,4 * n_classes ,            32 * n_classes)
                       
        self.nn5chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.nn5chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.nn5convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.nn5out = outconv(2 * n_classes, n_classes)
        
################################################################################################
################################################################################################
################################################################################################

        
        
        self.nn4upy4 = upd(1024,4 * n_classes)
        self.nn4encodery4x = n4encodery4x(1024,512,256,32,    32 * n_classes)
        #self.nn4encodery4 = nn4encodery4(1024 ,                     32 * n_classes)
                       
        self.nn4chy4e = ch(32 * n_classes  ,4 * n_classes)
        self.nn4chy4 = ch(4 * n_classes * 2 ,4 * n_classes)
        self.nn4convy4 = double_conv(4 * n_classes ,4 * n_classes)          
        ########################
        

        self.nn4upy3 = upd(4 * n_classes,3 * n_classes)
        self.nn4encodery3x = n4encodery3x(1024,512,256,32,    32 * n_classes)
        self.nn4encodery3 = n4encodery3(1024,       32 * n_classes)
                       
        self.nn4chy3e = ch(32 * n_classes * 2 ,3 * n_classes)
        self.nn4chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.nn4convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.nn4upy2 = upd(3 * n_classes,2 * n_classes)
        self.nn4encodery2x = n4encodery2x(1024,512,256,32,   32 * n_classes)
        self.nn4encodery2 = n4encodery2(1024,4 * n_classes ,            32 * n_classes)
                       
        self.nn4chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.nn4chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.nn4convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.nn4out = outconv(2 * n_classes, n_classes)

################################################################################################
################################################################################################
################################################################################################


        

        self.nn3upy3 = upd(512,3 * n_classes)
        self.nn3encodery3x = n3encodery3x(512,256,32,    32 * n_classes)
        #self.nn3encodery3 = nn3encodery3(512,256,       32 * n_classes)
                       
        self.nn3chy3e = ch(32 * n_classes  ,3 * n_classes)
        self.nn3chy3 = ch(3 * n_classes * 2 ,3 * n_classes)
        self.nn3convy3 = double_conv(3 * n_classes ,3 * n_classes)    
        ########################



        self.nn3upy2 = upd(3 * n_classes,2 * n_classes)
        self.nn3encodery2x = n3encodery2x(512,256,32,   32 * n_classes)
        self.nn3encodery2 = n3encodery2(512,            32 * n_classes)
                       
        self.nn3chy2e = ch(32 * n_classes * 2 ,2 * n_classes)
        self.nn3chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.nn3convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.nn3out = outconv(2 * n_classes, n_classes)
################################################################################################
################################################################################################
################################################################################################



        self.nn2upy2 = upd(256,2 * n_classes)
        self.nn2encodery2x = n2encodery2x(256,32,   32 * n_classes)
        #self.nn2encodery2 = nn2encodery2(512,            32 * n_classes)
                       
        self.nn2chy2e = ch(32 * n_classes  ,2 * n_classes)
        self.nn2chy2 = ch(2 * n_classes * 2 ,2 * n_classes)
        self.nn2convy2 = double_conv(2 * n_classes ,2 * n_classes)         
        ########################       

        #self.outc = outconv(64, n_classes) 
        self.nn2out = outconv(2 * n_classes, n_classes)
################################################################################################
################################################################################################
################################################################################################


        self.nn1convy1 = double_conv(32 ,2 * n_classes)  
        self.nn1out = outconv(2 * n_classes, n_classes)

################################################################################################
################################################################################################
################################################################################################


        self.nnout = outconv(6 * n_classes, n_classes)

    def forward(self, x):
        #print(x.size())
        #xx1 = x[1,:,:,:,:] 
        
        
        
        #print(xx1.size())
        
        
 
        
        
        x1 = self.inc(x)
        
        ########################
        x1d = self.down(x1)        
        x1e = self.encoderx1(x)
        x1c = torch.cat([x1d,x1e],dim=1)
        x1c = self.chx1(x1c)
        #x1c = x1d
               
        x2 = self.convx1(x1c)
        ########################
        x2d = self.down(x2)
        #print(x.size(),x1c.size())
        x2e = self.encoderx2(x,x1c)
        x2c = torch.cat([x2d,x2e],dim=1)  
        x2c = self.chx2(x2c)
        #x2c = x2d
             
        x3 = self.convx2(x2c)
        ########################
        x3d = self.down(x3)
        x3e = self.encoderx3(x,x1c,x2c)
        x3c = torch.cat([x3d,x3e],dim=1)   
        x3c = self.chx3(x3c) 
        #x3c = x3d
            
        x4 = self.convx3(x3c)        
    
        x4 = F.dropout(x4, p=0.5)
        ########################
        x4d = self.down(x4)
        x4e = self.encoderx4(x,x1c,x2c,x3c)        
        x4c = torch.cat([x4d,x4e],dim=1)  
        x4c = self.chx4(x4c) 
        #x4c = x4d
             
        x5 = self.convx4(x4c)     
        
        x5 = F.dropout(x5, p=0.5)
        ########################
        x5d = self.down(x5)
        x5e = self.encoderx5(x,x1c,x2c,x3c,x4c)        
        x5c = torch.cat([x5d,x5e],dim=1)  
        x5c = self.chx5(x5c) 
        #x5c = x5d
        
        x6 = x5c
        x6 = self.convx5(x5c)     
        x6 = F.dropout(x6, p=0.5)        
        
        #########################################################
        #########################################################
        #########################################################
        
        

        
        
        #########################################################
        #########################################################
        #########################################################




        n6y6 = x6
        
        
        n6y6p = self.n6upy6(n6y6)     
              
        n6y6e = self.n6encodery6x(x6,x5,x4,x3,x2,x1)
        n6y6e = self.n6chy6e(n6y6e)   
        #print(n6y6e.size(),n6y6p.size())        
        n6y6c = torch.cat([n6y6p,n6y6e],dim=1) 
        #print(y5c.size(),y5p.size(),y5e.size())
        n6y6c = self.n6chy6(n6y6c)
        n6y5 = self.n6convy6(n6y6c)        
        
        
        
        
                
        n6y5p = self.n6upy5(n6y5)     
              
        n6y5ex = self.n6encodery5x(x6,x5,x4,x3,x2,x1)
        n6y5ey = self.n6encodery5(n6y6)
        n6y5e = torch.cat([n6y5ex,n6y5ey],dim=1) 
        n6y5e = self.n6chy5e(n6y5e)   
        
        n6y5c = torch.cat([n6y5p,n6y5e],dim=1) 
        #print(y5c.size(),y5p.size(),y5e.size())
        n6y5c = self.n6chy5(n6y5c)
        n6y4 = self.n6convy5(n6y5c)
    
        
        
        n6y4p = self.n6upy4(n6y4)      
        
        n6y4ex = self.n6encodery4x(x6,x5,x4,x3,x2,x1)
        n6y4ey = self.n6encodery4(n6y6,n6y5)
        
        n6y4e = torch.cat([n6y4ex,n6y4ey],dim=1) 
        n6y4e = self.n6chy4e(n6y4e)        
        
        n6y4c = torch.cat([n6y4p,n6y4e],dim=1) 
        n6y4c = self.n6chy4(n6y4c)
        n6y3 = self.n6convy4(n6y4c)
        
        
        
        
        n6y3p = self.n6upy3(n6y3)      
        
        n6y3ex = self.n6encodery3x(x6,x5,x4,x3,x2,x1)
        n6y3ey = self.n6encodery3(n6y6,n6y5,n6y4)
        
        #print(n6y3ex.size(),n6y3ey.size())
        n6y3e = torch.cat([n6y3ex,n6y3ey],dim=1) 
        n6y3e = self.n6chy3e(n6y3e) 
        
        n6y3c = torch.cat([n6y3p,n6y3e],dim=1) 
        n6y3c = self.n6chy3(n6y3c)
        n6y2 = self.n6convy3(n6y3c)
        
        
        
        n6y2p = self.n6upy2(n6y2)     
        
        n6y2ex = self.n6encodery2x(x6,x5,x4,x3,x2,x1)
        n6y2ey = self.n6encodery2(n6y6,n6y5,n6y4,n6y3)
        
        #print(y2ex.size(),y2ey.size())        
        n6y2e = torch.cat([n6y2ex,n6y2ey],dim=1) 
        n6y2e = self.n6chy2e(n6y2e) 
        
        n6y2c = torch.cat([n6y2p,n6y2e],dim=1) 
        n6y2c = self.n6chy2(n6y2c)
        n6y1 = self.n6convy2(n6y2c)
        
        
        n6y = self.n6out(n6y1)        
        
    
   
   
        
        #########################################################
        #########################################################
        #########################################################
                    

        n5y5 = x5
                
        n5y5p = self.n5upy5(n5y5)     
              
        n5y5e = self.n5encodery5x(x5,x4,x3,x2,x1)
        n5y5e = self.n5chy5e(n5y5e)   
        
        n5y5c = torch.cat([n5y5p,n5y5e],dim=1) 
        #print(y5c.size(),y5p.size(),y5e.size())
        n5y5c = self.n5chy5(n5y5c)
        n5y4 = self.n5convy5(n5y5c)
    
        
        
        n5y4p = self.n5upy4(n5y4)      
        
        n5y4ex = self.n5encodery4x(x5,x4,x3,x2,x1)
        n5y4ey = self.n5encodery4(n5y5)
        
        n5y4e = torch.cat([n5y4ex,n5y4ey],dim=1) 
        n5y4e = self.n5chy4e(n5y4e)        
        
        n5y4c = torch.cat([n5y4p,n5y4e],dim=1) 
        n5y4c = self.n5chy4(n5y4c)
        n5y3 = self.n5convy4(n5y4c)
        
        
        
        
        n5y3p = self.n5upy3(n5y3)      
        
        n5y3ex = self.n5encodery3x(x5,x4,x3,x2,x1)
        n5y3ey = self.n5encodery3(n5y5,n5y4)
        

        n5y3e = torch.cat([n5y3ex,n5y3ey],dim=1) 
        n5y3e = self.n5chy3e(n5y3e) 
        
        n5y3c = torch.cat([n5y3p,n5y3e],dim=1) 
        n5y3c = self.n5chy3(n5y3c)
        n5y2 = self.n5convy3(n5y3c)
        
        
        
        n5y2p = self.n5upy2(n5y2)     
        
        n5y2ex = self.n5encodery2x(x5,x4,x3,x2,x1)
        n5y2ey = self.n5encodery2(n5y5,n5y4,n5y3)
        
             
        n5y2e = torch.cat([n5y2ex,n5y2ey],dim=1) 
        n5y2e = self.n5chy2e(n5y2e) 
        
        n5y2c = torch.cat([n5y2p,n5y2e],dim=1) 
        n5y2c = self.n5chy2(n5y2c)
        n5y1 = self.n5convy2(n5y2c)
        
        
        n5y = self.n5out(n5y1)


        #########################################################
        #########################################################
        #########################################################


        n4y4 = x4
        
        
        n4y4p = self.n4upy4(n4y4)      
        
        n4y4e = self.n4encodery4x(x4,x3,x2,x1)  
        n4y4e = self.n4chy4e(n4y4e)  
        
        n4y4c = torch.cat([n4y4p,n4y4e],dim=1) 
        #print(n4y4ex.size(),n4y4ey.size())
        n4y4c = self.n4chy4(n4y4c)
        n4y3 = self.n4convy4(n4y4c)
        
        
        
        
        n4y3p = self.n4upy3(n4y3)      
        
        n4y3ex = self.n4encodery3x(x4,x3,x2,x1)
        n4y3ey = self.n4encodery3(n4y4)
        
        #print(n4y3ex.size(),n4y3ey.size())
        n4y3e = torch.cat([n4y3ex,n4y3ey],dim=1) 
        n4y3e = self.n4chy3e(n4y3e) 
        
        n4y3c = torch.cat([n4y3p,n4y3e],dim=1) 
        n4y3c = self.n4chy3(n4y3c)
        n4y2 = self.n4convy3(n4y3c)
        
        
        
        n4y2p = self.n4upy2(n4y2)     
        
        n4y2ex = self.n4encodery2x(x4,x3,x2,x1)
        n4y2ey = self.n4encodery2(n4y4,n4y3)
        
        #print(y2ex.size(),y2ey.size())        
        n4y2e = torch.cat([n4y2ex,n4y2ey],dim=1) 
        n4y2e = self.n4chy2e(n4y2e) 
        
        n4y2c = torch.cat([n4y2p,n4y2e],dim=1) 
        n4y2c = self.n4chy2(n4y2c)
        n4y1 = self.n4convy2(n4y2c)
        
        
        n4y = self.n4out(n4y1)
        
        
        #########################################################
        #########################################################
        #########################################################


        n3y3 = x3
        
        
        n3y3p = self.n3upy3(n3y3)      
        
        n3y3e = self.n3encodery3x(x3,x2,x1)
        n3y3e = self.n3chy3e(n3y3e)        
        
        n3y3c = torch.cat([n3y3p,n3y3e],dim=1) 
        n3y3c = self.n3chy3(n3y3c)
        n3y2 = self.n3convy3(n3y3c)
        
        
        
        n3y2p = self.n3upy2(n3y2)     
        
        n3y2ex = self.n3encodery2x(x3,x2,x1)
        n3y2ey = self.n3encodery2(n3y3)
        
        #print(y2ex.size(),y2ey.size())        
        n3y2e = torch.cat([n3y2ex,n3y2ey],dim=1) 
        n3y2e = self.n3chy2e(n3y2e) 
        
        n3y2c = torch.cat([n3y2p,n3y2e],dim=1) 
        n3y2c = self.n3chy2(n3y2c)
        n3y1 = self.n3convy2(n3y2c)
        
        
        n3y = self.n3out(n3y1)        
        
        
        #########################################################
        #########################################################
        #########################################################


        n2y2 = x2
          
        
        n2y2p = self.n2upy2(n2y2)             
        n2y2e = self.n2encodery2x(x2,x1)
        n2y2e = self.n2chy2e(n2y2e)   
        #print(y2ex.size(),y2ey.size())        

        
        n2y2c = torch.cat([n2y2p,n2y2e],dim=1) 
        n2y2c = self.n2chy2(n2y2c)
        n2y1 = self.n2convy2(n2y2c)
        
        
        n2y = self.n2out(n2y1)            
        
        
        #########################################################
        #########################################################
        #########################################################


        n1y1 = x1
        n1y1 = self.n1convy1(n1y1)
        
        
        n1y = self.n1out(n1y1)   
        
        
        
        y = torch.cat([n6y,n5y,n4y,n3y,n2y,n1y],dim=1)
        y = self.nout(y)
        
        
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################        
        
        nn6y6 = x6
        
        
        nn6y6p = self.nn6upy6(nn6y6)     
              
        nn6y6e = self.nn6encodery6x(x6,x5,x4,x3,x2,x1)
        nn6y6e = self.nn6chy6e(nn6y6e)   
        #print(nn6y6e.size(),nn6y6p.size())        
        nn6y6c = torch.cat([nn6y6p,nn6y6e],dim=1) 
        #print(y5c.size(),y5p.size(),y5e.size())
        nn6y6c = self.nn6chy6(nn6y6c)
        nn6y5 = self.nn6convy6(nn6y6c)        
        
        
        
        
                
        nn6y5p = self.nn6upy5(nn6y5)     
              
        nn6y5ex = self.nn6encodery5x(x6,x5,x4,x3,x2,x1)
        nn6y5ey = self.nn6encodery5(nn6y6)
        nn6y5e = torch.cat([nn6y5ex,nn6y5ey],dim=1) 
        nn6y5e = self.nn6chy5e(nn6y5e)   
        
        nn6y5c = torch.cat([nn6y5p,nn6y5e],dim=1) 
        #print(y5c.size(),y5p.size(),y5e.size())
        nn6y5c = self.nn6chy5(nn6y5c)
        nn6y4 = self.nn6convy5(nn6y5c)
    
        
        
        nn6y4p = self.nn6upy4(nn6y4)      
        
        nn6y4ex = self.nn6encodery4x(x6,x5,x4,x3,x2,x1)
        nn6y4ey = self.nn6encodery4(nn6y6,nn6y5)
        
        nn6y4e = torch.cat([nn6y4ex,nn6y4ey],dim=1) 
        nn6y4e = self.nn6chy4e(nn6y4e)        
        
        nn6y4c = torch.cat([nn6y4p,nn6y4e],dim=1) 
        nn6y4c = self.nn6chy4(nn6y4c)
        nn6y3 = self.nn6convy4(nn6y4c)
        
        
        
        
        nn6y3p = self.nn6upy3(nn6y3)      
        
        nn6y3ex = self.nn6encodery3x(x6,x5,x4,x3,x2,x1)
        nn6y3ey = self.nn6encodery3(nn6y6,nn6y5,nn6y4)
        
        #print(nn6y3ex.size(),nn6y3ey.size())
        nn6y3e = torch.cat([nn6y3ex,nn6y3ey],dim=1) 
        nn6y3e = self.nn6chy3e(nn6y3e) 
        
        nn6y3c = torch.cat([nn6y3p,nn6y3e],dim=1) 
        nn6y3c = self.nn6chy3(nn6y3c)
        nn6y2 = self.nn6convy3(nn6y3c)
        
        
        
        nn6y2p = self.nn6upy2(nn6y2)     
        
        nn6y2ex = self.nn6encodery2x(x6,x5,x4,x3,x2,x1)
        nn6y2ey = self.nn6encodery2(nn6y6,nn6y5,nn6y4,nn6y3)
        
        #print(y2ex.size(),y2ey.size())        
        nn6y2e = torch.cat([nn6y2ex,nn6y2ey],dim=1) 
        nn6y2e = self.nn6chy2e(nn6y2e) 
        
        nn6y2c = torch.cat([nn6y2p,nn6y2e],dim=1) 
        nn6y2c = self.nn6chy2(nn6y2c)
        nn6y1 = self.nn6convy2(nn6y2c)
        
        
        nn6y = self.nn6out(nn6y1)        
        
    
   
   
        
        #########################################################
        #########################################################
        #########################################################
                    

        nn5y5 = x5
                
        nn5y5p = self.nn5upy5(nn5y5)     
              
        nn5y5e = self.nn5encodery5x(x5,x4,x3,x2,x1)
        nn5y5e = self.nn5chy5e(nn5y5e)   
        
        nn5y5c = torch.cat([nn5y5p,nn5y5e],dim=1) 
        #print(y5c.size(),y5p.size(),y5e.size())
        nn5y5c = self.nn5chy5(nn5y5c)
        nn5y4 = self.nn5convy5(nn5y5c)
    
        
        
        nn5y4p = self.nn5upy4(nn5y4)      
        
        nn5y4ex = self.nn5encodery4x(x5,x4,x3,x2,x1)
        nn5y4ey = self.nn5encodery4(nn5y5)
        
        nn5y4e = torch.cat([nn5y4ex,nn5y4ey],dim=1) 
        nn5y4e = self.nn5chy4e(nn5y4e)        
        
        nn5y4c = torch.cat([nn5y4p,nn5y4e],dim=1) 
        nn5y4c = self.nn5chy4(nn5y4c)
        nn5y3 = self.nn5convy4(nn5y4c)
        
        
        
        
        nn5y3p = self.nn5upy3(nn5y3)      
        
        nn5y3ex = self.nn5encodery3x(x5,x4,x3,x2,x1)
        nn5y3ey = self.nn5encodery3(nn5y5,nn5y4)
        

        nn5y3e = torch.cat([nn5y3ex,nn5y3ey],dim=1) 
        nn5y3e = self.nn5chy3e(nn5y3e) 
        
        nn5y3c = torch.cat([nn5y3p,nn5y3e],dim=1) 
        nn5y3c = self.nn5chy3(nn5y3c)
        nn5y2 = self.nn5convy3(nn5y3c)
        
        
        
        nn5y2p = self.nn5upy2(nn5y2)     
        
        nn5y2ex = self.nn5encodery2x(x5,x4,x3,x2,x1)
        nn5y2ey = self.nn5encodery2(nn5y5,nn5y4,nn5y3)
        
             
        nn5y2e = torch.cat([nn5y2ex,nn5y2ey],dim=1) 
        nn5y2e = self.nn5chy2e(nn5y2e) 
        
        nn5y2c = torch.cat([nn5y2p,nn5y2e],dim=1) 
        nn5y2c = self.nn5chy2(nn5y2c)
        nn5y1 = self.nn5convy2(nn5y2c)
        
        
        nn5y = self.nn5out(nn5y1)


        #########################################################
        #########################################################
        #########################################################


        nn4y4 = x4
        
        
        nn4y4p = self.nn4upy4(nn4y4)      
        
        nn4y4e = self.nn4encodery4x(x4,x3,x2,x1)  
        nn4y4e = self.nn4chy4e(nn4y4e)  
        
        nn4y4c = torch.cat([nn4y4p,nn4y4e],dim=1) 
        #print(nn4y4ex.size(),nn4y4ey.size())
        nn4y4c = self.nn4chy4(nn4y4c)
        nn4y3 = self.nn4convy4(nn4y4c)
        
        
        
        
        nn4y3p = self.nn4upy3(nn4y3)      
        
        nn4y3ex = self.nn4encodery3x(x4,x3,x2,x1)
        nn4y3ey = self.nn4encodery3(nn4y4)
        
        #print(nn4y3ex.size(),nn4y3ey.size())
        nn4y3e = torch.cat([nn4y3ex,nn4y3ey],dim=1) 
        nn4y3e = self.nn4chy3e(nn4y3e) 
        
        nn4y3c = torch.cat([nn4y3p,nn4y3e],dim=1) 
        nn4y3c = self.nn4chy3(nn4y3c)
        nn4y2 = self.nn4convy3(nn4y3c)
        
        
        
        nn4y2p = self.nn4upy2(nn4y2)     
        
        nn4y2ex = self.nn4encodery2x(x4,x3,x2,x1)
        nn4y2ey = self.nn4encodery2(nn4y4,nn4y3)
        
        #print(y2ex.size(),y2ey.size())        
        nn4y2e = torch.cat([nn4y2ex,nn4y2ey],dim=1) 
        nn4y2e = self.nn4chy2e(nn4y2e) 
        
        nn4y2c = torch.cat([nn4y2p,nn4y2e],dim=1) 
        nn4y2c = self.nn4chy2(nn4y2c)
        nn4y1 = self.nn4convy2(nn4y2c)
        
        
        nn4y = self.nn4out(nn4y1)
        
        
        #########################################################
        #########################################################
        #########################################################


        nn3y3 = x3
        
        
        nn3y3p = self.nn3upy3(nn3y3)      
        
        nn3y3e = self.nn3encodery3x(x3,x2,x1)
        nn3y3e = self.nn3chy3e(nn3y3e)        
        
        nn3y3c = torch.cat([nn3y3p,nn3y3e],dim=1) 
        nn3y3c = self.nn3chy3(nn3y3c)
        nn3y2 = self.nn3convy3(nn3y3c)
        
        
        
        nn3y2p = self.nn3upy2(nn3y2)     
        
        nn3y2ex = self.nn3encodery2x(x3,x2,x1)
        nn3y2ey = self.nn3encodery2(nn3y3)
        
        #print(y2ex.size(),y2ey.size())        
        nn3y2e = torch.cat([nn3y2ex,nn3y2ey],dim=1) 
        nn3y2e = self.nn3chy2e(nn3y2e) 
        
        nn3y2c = torch.cat([nn3y2p,nn3y2e],dim=1) 
        nn3y2c = self.nn3chy2(nn3y2c)
        nn3y1 = self.nn3convy2(nn3y2c)
        
        
        nn3y = self.nn3out(nn3y1)        
        
        
        #########################################################
        #########################################################
        #########################################################


        nn2y2 = x2
          
        
        nn2y2p = self.nn2upy2(nn2y2)             
        nn2y2e = self.nn2encodery2x(x2,x1)
        nn2y2e = self.nn2chy2e(nn2y2e)   
        #print(y2ex.size(),y2ey.size())        

        
        nn2y2c = torch.cat([nn2y2p,nn2y2e],dim=1) 
        nn2y2c = self.nn2chy2(nn2y2c)
        nn2y1 = self.nn2convy2(nn2y2c)
        
        
        nn2y = self.nn2out(nn2y1)            
        
        
        #########################################################
        #########################################################
        #########################################################


        nn1y1 = x1
        nn1y1 = self.nn1convy1(nn1y1)
        
        
        nn1y = self.nn1out(nn1y1)   
        
        
        
        yy = torch.cat([nn6y,nn5y,nn4y,nn3y,nn2y,nn1y],dim=1)
        yy = self.nnout(yy)        
        y = torch.sigmoid(y)
        return y, x5c



#####bottle####################################################################
class bottle_conv1(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(bottle_conv1, self).__init__()
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_ch, int(in_ch/4), 1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), int(in_ch/4), 3, padding=1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), in_ch, 1)
        )
        self.conv2 = nn.Sequential(
            
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_ch, int(in_ch/4), 1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), int(in_ch/4), 3, padding=1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), in_ch, 1),
            
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat([x,x,x,x,x,x,x,x],dim=1)
        x1 = self.conv1(x)
        x2 = x + x1
        x3 = self.conv2(x2)
        x4 = x2 + x3
        
        return x4

class bottle_conv2(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(bottle_conv2, self).__init__()
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_ch, int(in_ch/4), 1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), int(in_ch/4), 3, padding=1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), in_ch, 1)
        )
        self.conv2 = nn.Sequential(
            
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_ch, int(in_ch/4), 1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), int(in_ch/4), 3, padding=1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), in_ch, 1),
            
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat([x,x],dim=1)
        x1 = self.conv1(x)
        x2 = x + x1
        x3 = self.conv2(x2)
        x4 = x2 + x3
        
        return x4

class bottle_conv3(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(bottle_conv3, self).__init__()
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_ch, int(in_ch/4), 1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), int(in_ch/4), 3, padding=1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), in_ch, 1)
        )
        self.conv2 = nn.Sequential(
            
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_ch, int(in_ch/4), 1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), int(in_ch/4), 3, padding=1),
            
            nn.BatchNorm2d(int(in_ch/4)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(in_ch/4), in_ch, 1),
            
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x + x1
        x3 = self.conv2(x2)
        x4 = x2 + x3
        
        return x2








#####up####################################################################

class upd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upd, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch,  out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d( out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x      





class up1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up1, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d( out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x      
                
class up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up2, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch,  out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d( out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x          
        

class up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up3, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch,  2 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * out_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2* out_ch,   out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d( out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x


class up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up4, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch,  4 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * out_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4 * out_ch, 2 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * out_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2* out_ch,   out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d( out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x




class up5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up5, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 8 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8 * out_ch),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8 * out_ch, 4 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * out_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4 * out_ch, 2 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * out_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2* out_ch,   out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d( out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        #print(x.size())
        x = self.up(x)
        #print(x.size())
        return x




class up6(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up6, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 16 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16 * out_ch),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16 * out_ch, 8 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8 * out_ch),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8 * out_ch, 4 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * out_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4 * out_ch, 2 * out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * out_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2* out_ch,   out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d( out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        #print(x.size())
        x = self.up(x)
        #print(x.size())
        return x







#######in out#########################################################################

class out(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(out, self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1)
        )

    def forward(self, x1,x2,x3,x4,x5,x6):
        x = torch.cat([x1,x2,x3,x4,x5,x6],dim=1)
        x = self.conv(x)
        return x






class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True)
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
        self.conv = nn.Sequential(
            
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
                    
        return x
        


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
        
    
class encoderx5(nn.Module):
    def __init__(self, in_ch0, in_ch1, in_ch2, in_ch3,in_ch4, out_ch):
        super(encoderx5, self).__init__()
        self.down0 = nn.Sequential(
            nn.MaxPool2d(32),
            nn.Conv2d(in_ch0, int(out_ch/16), 1)
        )
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(16),
            nn.Conv2d(in_ch1, int(out_ch/16), 1)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(8),
            nn.Conv2d(in_ch2, int(out_ch/8), 1)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(in_ch3, int(out_ch/4), 1)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch4, int(out_ch/2), 1)
        )
        self.conv = single_conv(out_ch, out_ch)

    def forward(self, x0,x,x1c,x2c,x3c):
        x0 = self.down0(x0)
        x = self.down1(x)
        x1c = self.down2(x1c)
        x2c = self.down3(x2c)
        x3c = self.down4(x3c)        
        x = torch.cat([x0,x,x1c,x2c,x3c],dim=1)
        #x = self.conv(x)
        return x      
     
###########################################################################
###########################################################################    
###########################################################################










#####encoderyx##############################################################

class n6encodery6x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3, in_ch4, in_ch5,in_ch6, out_ch):
        super(n6encodery6x, self).__init__()
        
 
        
        self.up1 = nn.Sequential(
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
            nn.Conv2d(in_ch5, int(out_ch/32), 1),
            nn.MaxPool2d(8)
            
        )  
        
        self.down3 = nn.Sequential(
            nn.Conv2d(in_ch6, int(out_ch/32), 1),
            nn.MaxPool2d(16)
            
        ) 
    def forward(self, x6,x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size(),x6.size())
        x6 = self.up1(x6)
        x5 = self.ch(x5)
        x4 = self.down(x4)
        x3 = self.down1(x3)
        x2 = self.down2(x2)
        x1 = self.down3(x1)
        x = torch.cat([x1,x2,x3,x4,x5,x6],dim=1)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        return x        


class n6encodery5x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3, in_ch4, in_ch5,in_ch6, out_ch):
        super(n6encodery5x, self).__init__()
        
 
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/16), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )  
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )  
        self.ch = nn.Conv2d(in_ch3,int(out_ch/2),1)
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/8), 1),
            nn.MaxPool2d(2)
            
        )      
        
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch5, int(out_ch/32), 1),
            nn.MaxPool2d(4)
            
        )  
        
        self.down2 = nn.Sequential(
            nn.Conv2d(in_ch6, int(out_ch/32), 1),
            nn.MaxPool2d(8)
            
        )  
        

    def forward(self, x6,x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size(),x6.size())
        x6 = self.up2(x6)
        x5 = self.up1(x5)
        x4 = self.ch(x4)
        x3 = self.down(x3)
        x2 = self.down1(x2)
        x1 = self.down2(x1)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size(),x6.size())
        x = torch.cat([x1,x2,x3,x4,x5,x6],dim=1)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        return x  
                
class n6encodery4x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,in_ch5,in_ch6,out_ch):
        super(n6encodery4x, self).__init__()
        
        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/32), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )          
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/16), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    
                     
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )      
                  
        self.ch = nn.Conv2d(in_ch4,int(out_ch/2),1)
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch5, int(out_ch/8), 1),
            nn.MaxPool2d(2)
            
        )  
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch6, int(out_ch/32), 1),
            nn.MaxPool2d(4)
            
        )  
 
    def forward(self, x6,x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x6 = self.up3(x6)
        x5 = self.up2(x5)
        x4 = self.up1(x4)
        x3 = self.ch(x3)
        x2 = self.down(x2)
        x1 = self.down1(x1)
        
        x = torch.cat([x1,x2,x3,x4,x5,x6],dim=1)
        return x            
 
 
class n6encodery3x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,in_ch5,in_ch6,out_ch):
        super(n6encodery3x, self).__init__()
        
        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/32), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        ) 


        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/32), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )      
        
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/16), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        ) 
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        ) 
                  
        self.ch = nn.Conv2d(in_ch5,int(out_ch/2),1)
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch6, int(out_ch/8), 1),
            nn.MaxPool2d(2)
            
        )  
    def forward(self, x6,x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x6 = self.up3(x6)         
        x5 = self.up2(x5)
        x4 = self.up1(x4)
        x3 = self.up(x3)
        x2 = self.ch(x2)
        x1 = self.down(x1)

        x = torch.cat([x6,x5,x4,x3,x2,x1],dim=1)
        return x        
        
class n6encodery2x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3,in_ch4,in_ch5,in_ch6, out_ch):
        super(n6encodery2x, self).__init__()
        
        self.up4 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/32), 1),
            nn.Upsample(scale_factor=32, mode='bilinear')
            
        )         
        
        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/32), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        ) 
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/16), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        ) 
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/8), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        ) 
        
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch5, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )  
   
        self.ch = nn.Conv2d(in_ch6,int(out_ch/2),1)
    
    def forward(self, x6,x5,x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x6 = self.up4(x6)        
        x5 = self.up3(x5)
        x4 = self.up2(x4)
        x3 = self.up1(x3)
        x2 = self.up(x2)
        x1 = self.ch(x1)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x = torch.cat([x6,x5,x4,x3,x2,x1],dim=1)
        return x           
        
        
        
        
#####encodery##############################################################

class n6encodery5(nn.Module):
    def __init__(self, in_ch1, out_ch):
        super(n6encodery5, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, out_ch, 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )

    def forward(self, n6y5):
        x = self.up1(n6y5)
        return x 
        
        
        
class n6encodery4(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(n6encodery4, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/2), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/2), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
           
        )

    def forward(self, n6y5,n6y4):
        n6y5 = self.up1(n6y5)
        n6y4 = self.up2(n6y4)
        x = torch.cat([n6y5, n6y4],dim=1)

        return x  
        
class n6encodery3(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3 , out_ch):
        super(n6encodery3, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/4), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/2), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    

    def forward(self, n6y5, n6y4, n6y3):
        n6y5 = self.up1(n6y5)
        n6y4 = self.up2(n6y4)
        n6y3 = self.up3(n6y3)
        x = torch.cat([n6y5, n6y4, n6y3],dim=1)
        #x = self.conv(x)
        return x      


class n6encodery2(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3 , in_ch4 , out_ch):
        super(n6encodery2, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/8), 1),
            nn.Upsample(scale_factor=32, mode='bilinear')
            
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/8), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/4), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/2), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    

    def forward(self, n6y6, n6y5, n6y4, n6y3):
        n6y6 = self.up1(n6y6)    
        n6y5 = self.up2(n6y5)
        n6y4 = self.up3(n6y4)
        n6y3 = self.up4(n6y3)
        x = torch.cat([n6y6, n6y5, n6y4, n6y3],dim=1)
        #x = self.conv(x)
        return x 
 
###################################################################        
###################################################################           
###################################################################          
###################################################################           
###################################################################  




















###########################################################################
###########################################################################
###########################################################################
###########################################################################



#####encoderyx##############################################################

class n5encodery5x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3, in_ch4, in_ch5, out_ch):
        super(n5encodery5x, self).__init__()
        
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
        
class n5encodery4x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,in_ch5,out_ch):
        super(n5encodery4x, self).__init__()
        
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
 
 
class n5encodery3x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,in_ch5,out_ch):
        super(n5encodery3x, self).__init__()
        
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
        
class n5encodery2x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3,in_ch4,in_ch5, out_ch):
        super(n5encodery2x, self).__init__()
        
        
        
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
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x = torch.cat([x5,x4,x3,x2,x1],dim=1)
        return x           
        
        
        
        
#####encodery##############################################################

class n5encodery4(nn.Module):
    def __init__(self, in_ch1, out_ch):
        super(n5encodery4, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, out_ch, 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )

    def forward(self, n5y5):
        x = self.up1(n5y5)
        return x 
        
        
        
class n5encodery3(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(n5encodery3, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/2), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/2), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
           
        )

    def forward(self, n5y5,n5y4):
        n5y5 = self.up1(n5y5)
        n5y4 = self.up2(n5y4)
        x = torch.cat([n5y5, n5y4],dim=1)

        return x  
        
class n5encodery2(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3 , out_ch):
        super(n5encodery2, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/4), 1),
            nn.Upsample(scale_factor=16, mode='bilinear')
            
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/2), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    

    def forward(self, n5y5, n5y4, n5y3):
        n5y5 = self.up1(n5y5)
        n5y4 = self.up2(n5y4)
        n5y3 = self.up3(n5y3)
        x = torch.cat([n5y5, n5y4, n5y3],dim=1)
        #x = self.conv(x)
        return x      



 
###################################################################        
###################################################################           
###################################################################          
###################################################################           
###################################################################          
        


###########################################################################
###########################################################################
###########################################################################
###########################################################################



#####encoderyx##############################################################
   
        
class n4encodery4x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,out_ch):
        super(n4encodery4x, self).__init__()
        
                
             
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
            nn.Conv2d(in_ch4, int(out_ch/8), 1),
            nn.MaxPool2d(4)
            
        )   
    def forward(self, x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())

        x4 = self.up(x4)
        x3 = self.ch(x3)
        x2 = self.down(x2)
        x1 = self.down1(x1)
        
        x = torch.cat([x1,x2,x3,x4],dim=1)
        return x            
 
 
class n4encodery3x(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3,in_ch4,out_ch):
        super(n4encodery3x, self).__init__()
        
   
        
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/8), 1),
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
    def forward(self, x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())

        x4 = self.up1(x4)
        x3 = self.up(x3)
        x2 = self.ch(x2)
        x1 = self.down(x1)

        x = torch.cat([x4,x3,x2,x1],dim=1)
        return x        
        
class n4encodery2x(nn.Module):
    def __init__(self, in_ch1, in_ch2,in_ch3,in_ch4, out_ch):
        super(n4encodery2x, self).__init__()
        
        
        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch1, int(out_ch/8), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        ) 
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/8), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        ) 
        
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )  
   
        self.ch = nn.Conv2d(in_ch4,int(out_ch/2),1)
    
    def forward(self, x4,x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x4 = self.up2(x4)
        x3 = self.up1(x3)
        x2 = self.up(x2)
        x1 = self.ch(x1)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x = torch.cat([x4,x3,x2,x1],dim=1)
        return x           
        
        
        
        
#####encodery##############################################################
        
        
class n4encodery3(nn.Module):
    def __init__(self, in_ch2, out_ch):
        super(n4encodery3, self).__init__()

        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
           
        )

    def forward(self, n4y4):
        n4y4 = self.up2(n4y4)

        return n4y4 
        
class n4encodery2(nn.Module):
    def __init__(self, in_ch2, in_ch3 , out_ch):
        super(n4encodery2, self).__init__()

        
        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/2), 1),
            nn.Upsample(scale_factor=8, mode='bilinear')
            
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/2), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    

    def forward(self, n4y4, n4y3):

        n4y4 = self.up2(n4y4)
        n4y3 = self.up3(n4y3)
        x = torch.cat([ n4y4, n4y3],dim=1)
        #x = self.conv(x)
        return x      



 
###################################################################        
###################################################################           
###################################################################   


###########################################################################
###########################################################################
###########################################################################
###########################################################################



#####encoderyx##############################################################
            
 
 
class n3encodery3x(nn.Module):
    def __init__(self,  in_ch2, in_ch3,in_ch4,out_ch):
        super(n3encodery3x, self).__init__()
               
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        ) 
                  
        self.ch = nn.Conv2d(in_ch3,int(out_ch/2),1)
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch4, int(out_ch/4), 1),
            nn.MaxPool2d(2)
            
        )  
    def forward(self, x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())


        x3 = self.up(x3)
        x2 = self.ch(x2)
        x1 = self.down(x1)

        x = torch.cat([x3,x2,x1],dim=1)
        return x        
        
class n3encodery2x(nn.Module):
    def __init__(self,in_ch2,in_ch3,in_ch4, out_ch):
        super(n3encodery2x, self).__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(in_ch2, int(out_ch/4), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        ) 
        
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/4), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )  
   
        self.ch = nn.Conv2d(in_ch4,int(out_ch/2),1)
    
    def forward(self, x3,x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x3 = self.up1(x3)
        x2 = self.up(x2)
        x1 = self.ch(x1)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x = torch.cat([x3,x2,x1],dim=1)
        return x           
        
        
        
        
#####encodery##############################################################
        
        
class n3encodery2(nn.Module):
    def __init__(self,  in_ch3 , out_ch):
        super(n3encodery2, self).__init__()

        self.up3 = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch), 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
            
        )    

    def forward(self,  n3y3):


        n3y3 = self.up3(n3y3)

        #x = self.conv(x)
        return n3y3      



 
###################################################################        
###################################################################           
###################################################################  


###########################################################################
###########################################################################
###########################################################################
###########################################################################



#####encoderyx##############################################################
            
 
 
      
        
class n2encodery2x(nn.Module):
    def __init__(self,in_ch3,in_ch4, out_ch):
        super(n2encodery2x, self).__init__()
        
        
        self.up = nn.Sequential(
            nn.Conv2d(in_ch3, int(out_ch/2), 1),
            nn.Upsample(scale_factor=2, mode='bilinear')
            
        )  
   
        self.ch = nn.Conv2d(in_ch4,int(out_ch/2),1)
    
    def forward(self, x2,x1):
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())

        x2 = self.up(x2)
        x1 = self.ch(x1)
        #print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        x = torch.cat([x2,x1],dim=1)
        return x           
        
        
        
        
#####encodery##############################################################



 
###################################################################        
###################################################################           
################################################################### 