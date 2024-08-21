import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F

from  Gland_Segmentation.test_com import coms
from  Gland_Segmentation.test_cos import cosine 
from  Gland_Segmentation.get_image import get_file_extension 

import shutil



def mkdir(path): 
    folder = os.path.exists(path) 
    if not folder:                 
        os.makedirs(path)       
    else:
        shutil.rmtree(path) 
        os.makedirs(path)



def add_most_uc(iterr, net, new_train_files, pbatch, pcount, used_pixels, all_features, all_info, totalp, all_train, record, img_path, num_train ):

    
    ins = 32 * 6  
    batch_size = 1   
    imageType = 3  
      
    train_record_path = record + '/train/' + 'iter' + str(iterr-1) + '/' 
    train_image_path= img_path + '/train/img/'
    ext = get_file_extension(train_image_path)

    uc_map = []
    
    image_batch = np.zeros(batch_size*imageType* ins* ins)
    image_batch = image_batch.reshape( batch_size, imageType, ins, ins)
    
    for iduc in range(num_train):
        idT = 'c3_train_' + str(iduc + 1) + ext
        nameT = train_record_path + idT
        imgT = cv2.imread(nameT, cv2.IMREAD_GRAYSCALE)        
        uc_map.append(imgT)
        
        
    candidates = []
    cand_f = []
    for idp in range(pbatch * 2):
        print('############################getting:' + str(idp))
        
        maxv = 0 
        bestx = 0
        besty = 0
        bestid = 1
        bestusedp = 0 
        for idt in range(num_train):

            imgT = uc_map[idt]
            (w,h) = imgT.shape[:2]
            scale = 4
            for ii in range(int((w - ins)/scale)):
                for  jj in range(int((h - ins)/scale)):
                                        
                    i = int(scale*ii)
                    j = int(scale*jj)
                    
                    tempv = np.sum( imgT[i:i+ins, j:j+ins] )  
                    
                    
                    if tempv > maxv:
                        maxv = tempv
                        bestusedp = np.sum(  pcount[idt][i:i+ins, j:j+ins]  )
                        bestid = idt+1
                        bestx = i
                        besty = j
        
        i_dex = np.zeros(3).astype(int)
  
        i_dex[0] = int(bestid)
        i_dex[1] = int(bestx)
        i_dex[2] = int(besty)
      
        print('imageid:'+ str(i_dex[0]) + 'image_ij:' + str(i_dex[1]) + '_' +  str(i_dex[2])+ '    maxv:' + str(maxv) + '    usedpixel:' + str(bestusedp) )
        
        corr = np.zeros([ins,ins])        
        corr_imgT = uc_map[i_dex[0]-1]
        corr_imgT[i_dex[1]: i_dex[1]+ins, i_dex[2]: i_dex[2]+ins] = corr
        uc_map[i_dex[0]-1] = corr_imgT 
              
        candidates.append(i_dex)
    
        idC = 'train_' + str(i_dex[0]) + ext
        nameC = train_image_path + idC
        imgC = cv2.imread(nameC, cv2.IMREAD_COLOR)
        temp_imgC = imgC[i_dex[1]: i_dex[1]+ins, i_dex[2]: i_dex[2]+ins]
        temp_imgC = temp_imgC.transpose(2, 0, 1).astype(np.float64)
        
        for i in range(batch_size):
            image_batch[i,:,:,:] = temp_imgC        
        
        image_batch_t = torch.from_numpy(image_batch).float().cuda()        
        pred, xh = net(image_batch_t)        
        cand_f.append(xh.data.cpu().numpy())

    sim_m = np.zeros([len(cand_f), len(all_features)])
        
    for i1 in range(len(cand_f)):        
        for j1 in range(len(all_features)):                   
            cur_sim = cosine(cand_f[i1],all_features[j1])
            sim_m[i1][j1] = cur_sim
    
    max_score = -1000
    max_solu = []
            
            
    for i in range(len(coms)):        
        cur_score = 0        
        for i1 in range(len(all_features)):            
            max_sim = 0                    
            for j1 in range(pbatch):            
                if sim_m[  int(coms[i][j1])  ][i1] > max_sim :                    
                    max_sim = sim_m[  int(coms[i][j1])  ][i1]                                        
            cur_score = cur_score + max_sim * all_info[i1]   
        
        if cur_score > max_score:            
            max_score = cur_score
            max_solu = coms[i]
           
    print(max_score)
    print(max_solu)
            
    for ip in range(pbatch):    
        i_des = candidates[int(max_solu[ip])]
        
        bestid = i_des[0]
        bestx = i_des[1]
        besty = i_des[2]
        
        new_train_files.append(i_des)
            
        pc = pcount[i_des[0] - 1][int(i_des[1]): int(i_des[1]+ins), int(i_des[2]): int(i_des[2]+ins)]
        
        used_pixels = used_pixels + np.sum(pc)
        #print(np.sum(pc),used_pixels)
        pcount[i_des[0] - 1 ][int(i_des[1]): int(i_des[1]+ins), int(i_des[2]): int(i_des[2]+ins)] = np.zeros([ins,ins])
        
        train_image_path=img_path + '/train/img/'
        train_label_path=img_path +  '/train/mask/'
        
        record_dir = record + '/train_used/' + 'iter' + str(iterr) + '_'  

    
        idT = 'train_' + str(bestid ) + ext
        nameT = train_image_path + idT 
    
        imgT = cv2.imread(nameT, cv2.IMREAD_COLOR)

        imgT = imgT[bestx:bestx+ins,besty:besty+192,:]   
        cv2.imwrite(record_dir + 'c1_' + idT  , imgT  )

        maskExt = ext
        if 'bmp' in ext:
            maskExt = '_anno' + ext
        idL = 'train_' + str(bestid ) + maskExt
        nameL = train_label_path + idL
    
        imgL = cv2.imread(nameL, cv2.IMREAD_GRAYSCALE)

        imgL = imgL[bestx:bestx+ins,besty:besty+ins]   
    
        cv2.imwrite(record_dir + 'c2_' + idL  , imgL  )
        
        print('data load succesful') 
    return new_train_files, pcount, used_pixels
    
