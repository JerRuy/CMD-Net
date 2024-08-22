import numpy as np
import cv2
import os
import shutil

from Gland_Segmentation.pixel_count import pixelcount

def mkdir(path): 
    folder = os.path.exists(path) 
    if not folder:                 
        os.makedirs(path)       
    else:
        shutil.rmtree(path) 
        os.makedirs(path)

def random(pcount, used_pixels, ins,  img_path, record, ext):
    c = []
    c.append([2,48,90])
    c.append([24,61,59])
    c.append([44,24,127])
    c.append([46,53,193])
    c.append([48,14,99])
    c.append([53,60,107])    
    c.append([56,20,7])
    c.append([72,64,166])

    for i in range(len(c)):
        i_idx = c[i][0]
        stx = c[i][1]
        sty = c[i][2]
        pcount,used_pixels = temp(i_idx,stx,sty,pcount,used_pixels, ins, img_path, record, ext)
        
    return c, pcount, used_pixels


def temp(i_idx,stx,sty,pcount,used_pixels, ins, img_path, record, ext):
    train_image_path= img_path + '/train/img/'
    train_label_path= img_path + '/train/mask/'

    record_dir = record + '/train_used/' + 'iter0/'
    idT = 'train_' + str(i_idx) + ext
    nameT = train_image_path + idT 

    imgT = cv2.imread(nameT, cv2.IMREAD_COLOR)
    (w,h) = imgT.shape[:2]
    cv2.imwrite(record_dir + '/c1_' + idT  , imgT  )
    maskExt = ext
    if 'bmp' in ext:
        maskExt = '_anno' + ext

    idL = 'train_' + str(i_idx) + maskExt
    nameL = train_label_path + idL
    imgL = cv2.imread(nameL, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(record_dir + '/c2_' + idL  , imgL  )
    
    print(i_idx,stx,sty)
    temp_count = np.zeros([ins,ins])
    corr_count = pcount[int(i_idx-1)]
    used_pixels =  used_pixels + np.sum(corr_count[int(stx): int(stx+ins), int(sty): int(sty+ins)])
    print(np.shape(corr_count))
    print(np.shape(temp_count))
    print(stx,sty)
    corr_count[int(stx): int(stx+ins), int(sty): int(sty+ins)] = temp_count
    pcount[int(i_idx-1)] = corr_count
    return  pcount, used_pixels
