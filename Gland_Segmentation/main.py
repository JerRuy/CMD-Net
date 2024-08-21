import cv2
import sys
import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

#dataset selected
from Gland_Segmentation.random_add import random, mkdir
from Gland_Segmentation.add_most_uc import add_most_uc

#data loaded
from Gland_Segmentation.get_image import get_image
from Gland_Segmentation.pixel_count import pcountdis, pixelcount
from Gland_Segmentation.weight import initialize_weights

#test on valid and train
from Gland_Segmentation.eval_train import eval_train
from Gland_Segmentation.eval_test import eval_test



def train(net, batch_size, imageType, ins, lr, max_iter, img_path, record):
    current_p = 0
    init_iter = 0
    cur_lr = 0
    new_train_files = []
    meanerr = np.ones([100])

    addpoint_iter = 1000
    checkpoint_iter = 1000
    uc_th = 2000
    p_batch = 8

    pcount, totalp, num_train, ext = pixelcount(img_path, ins)
    initialize_weights(net)

    image_batch = np.zeros(batch_size*imageType* ins* ins)
    image_batch = image_batch.reshape( batch_size, imageType, ins, ins)
    traget_batch = np.zeros(batch_size* ins* ins)
    traget_batch = traget_batch.reshape(batch_size, ins, ins)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),
                        lr=lr,
                        weight_decay=0.0005)

    for iterr in range(init_iter,max_iter):
        if iterr % 100 == 0 :print(iterr)
        ###################################################################################################
        #add image to new_train_files

        if iterr % addpoint_iter == 0:
            net.eval()
            if iterr == 0:
                used_pixels = 0
                record_dir = record + '/train_used/iter0/'
                model_dir = record + '/model/'
                mkdir(record_dir) 
                mkdir(model_dir)
                new_train_files, pcount, used_pixels = random(pcount, used_pixels, ins, img_path, record, ext)
                pcountdis(num_train, pcount,totalp,used_pixels, ins)

            else:
                if (iterr >= uc_th) and (used_pixels <= totalp * 0.965): 
                    record_dir = record + '/train_used/iter' + str(iterr) + '/'  
                    mkdir(record_dir) 
                    new_train_files, pcount, used_pixels = add_most_uc(iterr, net, new_train_files, p_batch, pcount, used_pixels, all_features, all_info , totalp, all_train, record, img_path, num_train)
                
                pcountdis(num_train, pcount,totalp,used_pixels,ins)


        ######################################################################################################
        #model save and model test
        #train and test start from *999
        #train and No test start from *000
        net.eval()     
        if iterr  % checkpoint_iter == (checkpoint_iter - 2):
            eval_test(iterr,net, img_path, record, ins)
            
        if ( iterr  % checkpoint_iter == (checkpoint_iter - 1) ) and (iterr  >=  (uc_th-1))  :            
            all_features, all_info, all_train = eval_train(iterr,net,pcount, img_path, record, ins, num_train)               
            
        if iterr  % checkpoint_iter == (checkpoint_iter - 2): 
            if iterr > (uc_th -2):            
                print('save state')
                state = {'epoch': iterr,
                'model_state': net.state_dict(),
                'optimizer_state': optimizer.state_dict(), 
                'used_pixels' :used_pixels,
                'totalp'  :totalp,
                'pcount'  :pcount,
                'all_features' : all_features,
                'all_info': all_info,
                'new_train_files': new_train_files}
                torch.save(state, record + '/model/'+str(iterr)+'.pkl')
            else:             
                print('save state')
                state = {'epoch': iterr,
                'model_state': net.state_dict(),
                'optimizer_state': optimizer.state_dict(), 
                'used_pixels' :used_pixels,
                'totalp'  :totalp,
                'pcount'  :pcount,
                'new_train_files': new_train_files}
                torch.save(state, record+ '/model/'+str(iterr)+'.pkl')
        
        ######################################################################################################
        #load_image and train
        net.train()
        
        for i in range(batch_size):
            tr_sp, la_sp = get_image(new_train_files, img_path, ins, ext)
            tr_sp = tr_sp.transpose(2, 0, 1)
            la_sp[la_sp>0]=1
            image_batch[i,:,:,:] = tr_sp.astype(np.float64)
            traget_batch[i,:,:] = la_sp.astype(np.float64)

                
        image_batch_t=torch.from_numpy(image_batch).float()
        #print(image_batch_t.type())
        image_batch_t=image_batch_t.cuda()
        target_batch_ts=torch.from_numpy(traget_batch).long()
        target_batch_ts=target_batch_ts.cuda()

        pred, _ = net(image_batch_t)
        loss = criterion(pred, target_batch_ts) 
        
        current_p = current_p + 1
        if current_p > 99 : current_p = 1
        meanerr[current_p] = loss.data.item()

        print('Iteration: ' + str(iterr) + ' Loss: ' + str(np.around(np.mean(meanerr), decimals=3))  + ' lr: ' + str(cur_lr)  +        ' used_pixel: ' + str(used_pixels/totalp) ) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        ######################################################################################################
        #learning rate

        for param_group in optimizer.param_groups:
            if  iterr < max_iter * 0.1:
                param_group['lr'] = lr 
            else:
                param_group['lr'] = lr * 0.1 
            cur_lr =  param_group['lr']   

def CMD_Net_GS_Test(net, model_path, test_path, record_path, ins, gpuid):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpuid))
    net.load_state_dict(checkpoint['model_state'])
    print('succesful loaded')
    eval_test(0,net, test_path, record_path, ins)


