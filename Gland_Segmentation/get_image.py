import numpy as np
import cv2
import os

np.set_printoptions(threshold = 1e6)



def get_image(new_train_files, img_path, ins, ext):
    random_id = np.random.randint(0,len(new_train_files))    
 
      
    train_image_path= img_path + '/train/img/'
    train_label_path= img_path + '/train/mask/'
      
    # print(new_train_files[random_id])
    image_id = int(new_train_files[random_id][0])
    stx = int(new_train_files[random_id][1]) 
    sty = int(new_train_files[random_id][2])
    #print(image_id,stx,sty)

    
    image_name = train_image_path + 'train_' + str(image_id) + ext
    maskExt = ext
    if 'bmp' in ext:
        maskExt = '_anno' + ext
    label_name = train_label_path + 'train_' + str(image_id) + maskExt
    #print(image_name)
    
    train_sample = cv2.imread(image_name, cv2.IMREAD_COLOR)
    label_sample = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)


    train_sample = train_sample[stx:stx+ins, sty:sty+ins,:]
    label_sample = label_sample[stx:stx+ins, sty:sty+ins]
    
    flipi = np.random.randint(1,2+1)  
    rotate = np.random.randint(1,4+1)   

    
    (w,h) = train_sample.shape[:2]
    center =  ( int(h/2), int(w/2) )
    M = cv2.getRotationMatrix2D(center,90*rotate,1)
    if rotate == 4:
        #print('rotate'+ str(rotate))
        train_sample = train_sample 
        label_sample = label_sample 
    else:
        #print('rotate'+ str(rotate))
        train_sample = cv2.warpAffine(train_sample,M,(w,h))
        label_sample = cv2.warpAffine(label_sample,M,(w,h))
    
    
    if flipi == 1:
        #print('flip')
        train_sample = cv2.flip(train_sample, 1)
        label_sample = cv2.flip(label_sample, 1)      
    
    
    return train_sample, label_sample


def get_file_extension(directory):
    file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    first_file_extension = None
    if file_list:
        first_file = file_list[0]
        _, first_file_extension = os.path.splitext(first_file)
    return first_file_extension

if __name__ == '__main__':



    c = []
    c.append([1,2,3])
    c.append([21,22,3])
    c.append([11,2,35])
    
    
    t, p ,q = get_image(c)    
      