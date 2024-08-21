## __CMDNet__ : CMD Net based on pytorch
### Introduction
This project code based on python and pytorch framework, including data preprocessing, model training and testing, visualization, etc. This project is suitable for researchers who study CMD Net.  
![image](https://github.com/JerRuy/CMD-Net/blob/main/img/pic.png)


### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.9.19                    
pytorch                   2.3.1
pytorch-cuda              11.8
torchvision               0.18.1
numpy                     1.26.4        
opencv                    4.10.0.84        
ipython                   8.15.0       
scipy                     1.13.1       
scikit-learn              1.4.2          
pillow                    10.4.0      
```  
The above environment is successful when running the code of the project. 
__It is recommended that you use Linux for training and testing__

## Usage 
### Project Structure

The project structure are as follows : 
```
Project
  ├── Gland_Segmentation
  │   ├── add_most_uc.py
  │   ├── eval_test.py
  │   ├── eval_train.py
  │   ├── get_image.py
  │   ├── main.py
  │   ├── pixel_count.py
  │   ├── random_add.py
  │   ├── test_com.py
  │   ├── test_cos.py
  │   └── weight.py
  ├── main.py
  ├── networks
  │   ├── CMDNet_ConvUNeXt.py
  │   ├── CMDNet_DeepLab.py
  │   ├── CMDNet_FCN.py
  │   ├── CMDNet_FCNsa2.py
  │   ├── CMDNet_FCNsa.py
  │   ├── CMDNet_SegNet.py
  │   ├── CMDNet_UNet.py
  │   ├── ConvUNeXt.py
  │   ├── DeepLab.py
  │   ├── FCN.py
  │   ├── FCNsa.py
  │   ├── SegNet.py
  │   ├── UNet.py
  │   └── utils
  │       ├── CMDNet_DeepLab_res.py
  │       ├── CMDNet_DeepLab_res_utils.py
  │       └── CMDNet_DeepLab_utils.py
  ├── README.md
  ├── scripts
  │   ├── data.py
  │   ├── huffmancoding.py
  │   ├── __init__.py
  │   ├── loss.py
  │   └── quant.py
  ├── Vessel_Segmentation
  │   ├── test.py
  │   └── train.py
  ├── dataset
  │   ├── CHASEDB1
  │   │   ├── test
  │   │   │   ├── images
  │   │   │   ├── labels1
  │   │   │   └── labels2
  │   │   └── training
  │   │       ├── images
  │   │       ├── labels1
  │   │       └── labels2
  │   ├── CRAG
  │   │   ├── test
  │   │   │   ├── img
  │   │   │   └── mask
  │   │   ├── train
  │   │   │   ├── img
  │   │   │   └── mask
  │   │   └── valid
  │   │       └── Images
  │   ├── DRIVE
  │   │   ├── test
  │   │   │   ├── 1st_manual
  │   │   │   ├── 2nd_manual
  │   │   │   ├── images
  │   │   │   └── mask
  │   │   └── training
  │   │       ├── 1st_manual
  │   │       └── images
  │   └── GlaS
  │       ├── testA
  │       ├── testA_masks
  │       ├── testB
  │       ├── testB_masks
  │       └── train
  │           ├── img
  │           └── mask
  └── outcome
      ├── CHASEDB1/
      ├── CRAG/
      ├── DRIVE/
      └── GlaS/
```
### Datasets preparation 
1. Please download the retina image datasets from the official address.  
2. Unzip the downloaded datasets file. The directory tree are as follows:  
```
  datasets
    ├── CHASEDB1
    │   ├── test
    │   │   ├── images
    │   │   ├── labels1
    │   │   └── labels2
    │   └── training
    │       ├── images
    │       ├── labels1
    │       └── labels2
    ├── CRAG
    │   ├── test
    │   │   ├── img
    │   │   └── mask
    │   ├── train
    │   │   ├── img
    │   │   └── mask
    │   └── valid
    │       └── Images
    ├── DRIVE
    │   ├── test
    │   │   ├── 1st_manual
    │   │   ├── 2nd_manual
    │   │   ├── images
    │   │   └── mask
    │   └── training
    │       ├── 1st_manual
    │       └── images
    └── GlaS
        ├── testA
        ├── testA_masks
        ├── testB
        ├── testB_masks
        └── train
            ├── img
            └── mask

```

### Parameters
Below is a description of the parameters and their default values:

* --phase: Specifies the phase of operation. Options are train and test. Default is train.
* --batch_size: The batch size for training. Default is 1.
* --dataset: The dataset of the input images. Options are CHASEDB, CRAG, DRIVE, and GlaS. Default is DRIVE.
* --image_path: The size of the input images. Default is ./dataset/DRIVE.
* --model: The name of model. Options are CMDNet_ConvUNeXt, CMDNet_DeepLab, CMDNet_FCN, CMDNet_FCNsa, CMDNet_SegNet, CMDNet_UNet, ConvUNeXt, DeepLab, FCN, FCNsa, SegNet, UNet. Default is  FCN.
* --save-path: The directory where the train model checkpoints will be saved.
* --epoch: The number of epochs for training. Default is 54000.
* --n-channels: The numbers of channels.  Default is 3.
* --n-classes: The numbers of classes.  Default is 2.
* --lr: The learning rate for the optimizer. Default is 2e-4.
* --gpuid: The id of gpu. Default is 0.
* --early-stop: The early of stop. Default is 20.
* --update-lr: The learning rate for update. Default is 10.
* --epochloss_init: The epoch loss used to init. Default is 10000

  
### Training model
To train the model, supported datasets include: CHASEDB, CRAG, DRIVE, and GlaS and supported models include: CMDNet_ConvUNeXt, CMDNet_DeepLab, CMDNet_FCN, CMDNet_FCNsa2, CMDNet_FCNsa, CMDNet_SegNet, CMDNet_UNet, ConvUNeXt, DeepLab, FCN, FCNsa, SegNet, UNet. run command as following :

```
python main.py  --phase 'train' \
                --dataset 'DRIVE' \
                --input-size 448 \
                --model 'CMDNet_UNet' \
                --save-path 'outcome/DRIVE/' \
                --alpha 0.0001 \
                --epoch 45000 \
                --batchsize 4 \
                --n-channels 3 \
                --n-classes 1 \
                --lr 2e-4 \
                --gpuid 0 \
                --early-stop 20 \
                --update-lr 10 \
                --epochloss_init 10000
```


### Testing model
To test the model, supported datasets include: CHASEDB, CRAG, DRIVE, and GlaS and supported models include: CMDNet_ConvUNeXt, CMDNet_DeepLab, CMDNet_FCN, CMDNet_FCNsa2, CMDNet_FCNsa, CMDNet_SegNet, CMDNet_UNet, ConvUNeXt, DeepLab, FCN, FCNsa, SegNet, UNet. run command as following :
```
python main.py  --phase 'test'  \
                --dataset 'DRIVE' \
                --input-size 448  \
                --model 'CMDNet_UNet' \
                --model-path './outcome/DRIVE/CMD-Net_DRIVE.th' \
                --alpha 0.0001 \
                --batchsize 4 \
                --n-channels 3 \
                --n-classes 1 \
                --gpuid 0 

```  
