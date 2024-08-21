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


# def random_add(pcount, used_pixels, ins, img_path, record):

#     num_train = 85
#     ins = 32 * 6  
      
      
      
      
#     train_image_path= img_path + '/train/img/'
#     train_label_path= img_path + '/train/mask/'
      
      
#     i_idx = np.random.randint(1,num_train + 1)
      
#     listing = os.listdir(train_image_path)
#     img_name = train_image_path + 'train_' + str(i_idx) + '.bmp'
#     image = cv2.imread(img_name)
    
#     [xs,ys,zs] = np.shape(image)
#     #print(img_name)
#     #print(np.shape(image))  
#     #print(xs,ys,zs)   
    
#     xss = xs
#     yss = ys
      
#     stx = np.random.randint(0,xss-ins )  
#     sty = np.random.randint(0,yss-ins )    
      
#     i_dex = np.zeros(3)
    
#     i_dex[0] = int(i_idx)
#     i_dex[1] = int(stx)
#     i_dex[2] = int(sty)

#     record_dir = record + '/train_used/' + 'iter0/'
    
#     idT = 'train_' + str(i_idx) + '.bmp'
#     nameT = train_image_path + idT 

#     imgT = cv2.imread(nameT, cv2.IMREAD_COLOR)
#     (w,h) = imgT.shape[:2]
        

#     cv2.imwrite(record_dir + '/c1_' + idT  , imgT  )


#     idL = 'train_' + str(i_idx) + '.bmp'
#     nameL = train_label_path + idL

#     imgL = cv2.imread(nameL, cv2.IMREAD_GRAYSCALE)
    


#     cv2.imwrite(record_dir + '/c2_' + idL  , imgL  )
    
#     #print(i_dex)
#     print(img_name,stx,sty)
#     temp_count = np.zeros([ins,ins])
    
#     corr_count = pcount[int(i_dex[0]-1)]
    
#     used_pixels =  used_pixels + np.sum(corr_count[int(i_dex[1]): int(i_dex[1]+ins), int(i_dex[2]): int(i_dex[2]+ins)])
    
#     corr_count[int(i_dex[1]): int(i_dex[1]+ins), int(i_dex[2]): int(i_dex[2]+ins)] = temp_count
#     pcount[int(i_dex[0]-1)] = corr_count        
#     return i_dex, pcount, used_pixels
    
      
      
if __name__ == '__main__':
    
    ins =192

    pcount = pixelcount()
    new_image,pcount = random_add(pcount)   
      
    print(new_image[0],new_image[1],new_image[2])  
    print(pcount[ int(new_image[0] -1) ])
    print(pcount[int(new_image[0] -1)][int(new_image[1]): int(new_image[1]+ins), int(new_image[2]): int(new_image[2]+ins)] )  
      