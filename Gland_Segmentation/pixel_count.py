import numpy as np
import cv2
import os
from  Gland_Segmentation.get_image import get_file_extension 

def pixelcount(img_path, ins):
    train_image_path=img_path + '/train/img/'
    train_label_path=img_path + '/train/mask/'
    num_train = len(os.listdir(train_image_path)) -1
    ext = get_file_extension(train_image_path)
    listing = os.listdir(train_image_path)
    pcount = []
    totalp = 0
    for i_idx in range(num_train):
        img_name = train_image_path + 'train_' + str(i_idx+1) + ext
        image = cv2.imread(img_name)
        [xs,ys,zs] = np.shape(image)
        xss = xs
        yss = ys
        temp_count = np.ones([xss,yss])
        pcount.append(temp_count)
        totalp = totalp + np.sum(temp_count) 
    return pcount, totalp, num_train, ext

def pcountdis(num_train, pcount,totalp, used_pixels, ins):
    temp_pixelcount = 0
    for iii in range(num_train):
        temp_sum = np.sum(pcount[iii]) 
        [xs,ys] = np.shape(pcount[iii])
        temp_pixelcount = temp_pixelcount + np.sum(pcount[iii]) 
        print(iii+1,(xs*ys - temp_sum ) /ins /ins)
        
    print(totalp - temp_pixelcount)
    print(used_pixels)  

