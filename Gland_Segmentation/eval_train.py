import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from network.unet_combination_o import UNet_combination_o_f
import os
import shutil
from  Gland_Segmentation.get_image import get_file_extension 

def softmax02(x):
    x = nn.Softmax2d()(x)
    #x = F.softmax(x)
    x = x * 0.2
    return x

def softmax03(x11,x12,x21,x22,x31,x32,x41,x42,x51,x52):
    xx11 = softmax02(x11)
    xx21 = softmax02(x21)
    xx31 = softmax02(x31)
    xx41 = softmax02(x41)
    xx51 = softmax02(x51)

    xx12 = softmax02(x12)
    xx22 = softmax02(x22)
    xx32 = softmax02(x32)
    xx42 = softmax02(x42)
    xx52 = softmax02(x52)

        
    return xx11,xx12,xx21,xx22,xx31,xx32,xx41,xx42,xx51,xx52
    
 
    
    
    
def  rotate02(temp_train, weight, batch_size, image_batch, net):
    
    temp_train = temp_train.transpose(2, 0, 1).astype(np.float64)
    
    # for i in range(batch_size):
    #         #print(np.shape(temp_train))
    #     image_batch[i,:,:,:] = temp_train
    for i in range(batch_size):
        image_batch[i,:,:,:] = temp_train
    
    image_batch_t = torch.from_numpy(image_batch).float().cuda()
    
    pred, xh = net(image_batch_t)
        
    xh = softmax02(xh)
    xxx_std = torch.std(pred)
    temp_pred_s = pred.data.cpu().numpy()
    
    return temp_pred_s, xxx_std, xh
    
    



def mkdir(path): 
    folder = os.path.exists(path) 
    if not folder:                 
        os.makedirs(path)       
    else:
        shutil.rmtree(path) 
        os.makedirs(path)
    
    
    
def eval_train(iterr, net, pcount, img_dir, record, ins,  num_train):    
    test_dir = img_dir + '/train/img/'
    record_dir = record + '/train/' + 'iter' + str(iterr) + '/'
    ext = get_file_extension(test_dir)
    
    mkdir(record_dir)
       
    weight = np.zeros([ins,ins])
    for i in range(ins):
        for j in range(ins):
            dx = min(i, ins - i)
            dy = min(j, ins - j)   
            d = min(dx,dy)
            weight[i,j] = d+1
    weight = weight/(weight.max())  
    cv2.imwrite("./weight.png", weight) 
    
    batch_size = 1
    imageType = 3
    image_batch = np.zeros(batch_size*imageType* ins* ins)
    image_batch = image_batch.reshape( batch_size, imageType, ins, ins)
    
    
    
    
    all_features = []
    all_info = []
    all_train = []
    
    

    for i in range(num_train):    
        idT = 'train_' + str(i + 1) + ext
        nameT = test_dir + idT
        print(nameT)
        imgT = cv2.imread(nameT, cv2.IMREAD_COLOR)

        (ww,hh) = imgT.shape[:2]
        temp_pcount = pcount[i] 
        avk = 2
        
        big_pred_s = np.zeros([ww,hh])
        big_pred_b = np.zeros([ww,hh])
        big_xxx_std = np.zeros([ww,hh])
        big_weight = np.zeros([ww,hh])
        #print(np.shape(big_pred_s))
                
        for i1 in range( math.ceil(   (ww - ins) / (ins / avk) ) +1 ):
            for j1 in range( math.ceil(   (hh - ins) / (ins / avk) ) +1 ):
                
                insi = int( i1 * ins / avk )
                insj = int( j1 * ins / avk )
                
                insdi = insi + ins
                insdj = insj + ins

                if insdi > ww:
                    insi = ww - ins
                    insdi = ww
                
                if insdj > hh:
                    insj = hh - ins
                    insdj = hh
                
                
                print(insi,insj,insdi,insdj)

                temp_train = imgT[insi:insdi,insj:insdj,:]   
                
                temp_count = temp_pcount[insi:insdi,insj:insdj]
                temp_pred_s, xxx_std, xh = rotate02(temp_train, weight, batch_size, image_batch, net)
                temp_pred_s = temp_pred_s[0,1,:,:]	
                big_pred_s[insi:insdi,insj:insdj] = big_pred_s[insi:insdi,insj:insdj] + temp_pred_s * weight
                big_xxx_std[insi:insdi,insj:insdj] = big_xxx_std[insi:insdi,insj:insdj] + xxx_std.data.cpu().numpy() *temp_count * weight *10
                
                big_weight[insi:insdi,insj:insdj] = big_weight[insi:insdi,insj:insdj] + weight
                
                all_features.append(xh.data.cpu().numpy())
                all_info.append(np.sum(temp_count)/ins/ins)

        
        final_pred_s = big_pred_s / big_weight *255
        final_xxx_std = big_xxx_std / big_weight *255
        cv2.imwrite(record_dir + '/c1_' + idT  , final_pred_s )
        cv2.imwrite(record_dir + '/c3_' + idT   , final_xxx_std )


        all_train.append(final_xxx_std)

    return all_features, all_info, all_train

    
    
    
    
    