import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import shutil
import sys


def softmax02(x):
    x = nn.Softmax2d()(x)
    #x = F.softmax(x)
    x = x * 0.2
    return x

    
def  rotate02(temp_train, batch_size, image_batch, net):
    
    temp_train = temp_train.transpose(2, 0, 1).astype(np.float64)
    
    for i in range(batch_size):
            #print(np.shape(temp_train))
            image_batch[i,:,:,:] = temp_train
    
    
    image_batch_t = torch.from_numpy(image_batch).float().cuda()
    pred, _ = net(image_batch_t)
    return softmax02(pred)
    
    
def mkdir(path): 
    folder = os.path.exists(path) 
    if not folder:                 
        os.makedirs(path)       
    else:
        shutil.rmtree(path) 
        os.makedirs(path)
    
    
    
def has_subdirectories(directory):
    items = os.listdir(directory)
    
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            return True
    
    return False
    
def eval_test(iterr, net, test_dir, record, ins):
    
    record_dir = record + '/test/' + 'iter' + str(iterr) + '/'
    mkdir(record_dir)
    
    weight = np.zeros([ins,ins])
    for i in range(ins):
        for j in range(ins):
            dx = min(i, ins - i)
            dy = min(j, ins - j)   
            d = min(dx,dy)
            weight[i,j] = d + 1
    weight = weight/(weight.max())  
    cv2.imwrite("./weight.png", weight) 
    

    batch_size = 1
    imageType = 3
    n_class  = 2
    image_batch = np.zeros(batch_size*imageType* ins* ins)
    image_batch = image_batch.reshape( batch_size, imageType, ins, ins)
    for item in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item)
        if "test" in item and "mask" not in item and os.path.isdir(item_path):            
            if has_subdirectories(item_path):
                item_path = item_path + "/img"
            for idA in os.listdir(item_path):
                nameA = os.path.join(item_path, idA)
                print(nameA)
                imgA = cv2.imread(nameA, cv2.IMREAD_COLOR)
                
                (ww,hh) = imgA.shape[:2]
                
                avk = 4
                big_pred_s = np.zeros([ww,hh])
                big_pred_b = np.zeros([ww,hh])
                        
                
                big_weight = np.zeros([ww,hh])
                        
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
                            
                        
                        temp_train = imgA[insi:insdi,insj:insdj,:]   
                        
                        center = (ins/2 , ins/2 )
                        M1 = cv2.getRotationMatrix2D(center,90*1,1)
                        M2 = cv2.getRotationMatrix2D(center,90*2,1)
                        M3 = cv2.getRotationMatrix2D(center,90*3,1)
                        
                        MM1 = cv2.getRotationMatrix2D(center,-90*1,1)
                        MM2 = cv2.getRotationMatrix2D(center,-90*2,1)
                        MM3 = cv2.getRotationMatrix2D(center,-90*3,1)
                        
                        
                        temp_train1 = cv2.warpAffine(temp_train,M1,(ins,ins))
                        temp_train2 = cv2.warpAffine(temp_train,M2,(ins,ins))
                        temp_train3 = cv2.warpAffine(temp_train,M3,(ins,ins))         
                        temp_train4 = temp_train
                        
                        temp_pred_s1 = rotate02(temp_train1,  batch_size,  image_batch, net)         
                        temp_pred_s2 = rotate02(temp_train2,  batch_size,  image_batch, net)    
                        temp_pred_s3 = rotate02(temp_train3,  batch_size,  image_batch, net)    
                        temp_pred_s4 = rotate02(temp_train4,  batch_size,  image_batch, net)    
                        
                        results = []
                        temp_pred_s1 = cv2.warpAffine(temp_pred_s1[0,1,:,:].cpu().detach().numpy(),MM1,(ins,ins))
                        results.append(temp_pred_s1)
                        temp_pred_s2 = cv2.warpAffine(temp_pred_s2[0,1,:,:].cpu().detach().numpy(),MM2,(ins,ins))
                        results.append(temp_pred_s2)
                        temp_pred_s3 = cv2.warpAffine(temp_pred_s3[0,1,:,:].cpu().detach().numpy(),MM3,(ins,ins))
                        results.append(temp_pred_s3)                        
                        
                        temp_pred_s4 = temp_pred_s4[0,1,:,:].cpu().detach().numpy()
                        results.append(temp_pred_s4)                        

                        temp_pred_s = np.mean(results, axis=0) 
                        
                        big_pred_s[insi:insdi,insj:insdj] = big_pred_s[insi:insdi,insj:insdj] + temp_pred_s * weight
                        big_weight[insi:insdi,insj:insdj] = big_weight[insi:insdi,insj:insdj] + weight

                
                final_pred_s = big_pred_s / big_weight                         
                final_pred_s = final_pred_s *255                

                cv2.imwrite(record + '/test/' + 'iter' + str(iterr) + '/c1_' + idA  , final_pred_s )