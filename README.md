## __CMDNet__ : CMD Net based on pytorch
### Introduction
This project code based on python and pytorch framework, including data preprocessing, model training and testing, visualization, etc. This project is suitable for researchers who study CMD Net.  


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
### Training model
To train the model by DRIVE data, run the following command:

```
python main.py  --phase 'train' \
                --input-size 448 \
                --image-path './dataset/DRIVE' \
                --model 'FCNsa' \
                --save-path 'weights/1/' \
                --alpha 0.0001 \
                --epoch 300 \
                --batchsize 4 \
                --n-channels 3 \
                --n-classes 1 \
                --lr 2e-4 \
                --gpuid 0 \
                --early-stop 20 \
                --update-lr 10 \
                --epochloss_init 10000
```

To train the model by CHASEDB data, run the following command:

```
python main.py  --phase 'train' \
                --input-size 448 \
                --image-path './dataset/CHASEDB1' \
                --model 'FCNsa' \
                --save-path 'weights/2/' \
                --alpha 0.0001 \
                --epoch 300 \
                --batchsize 4 \
                --n-channels 3 \
                --n-classes 1 \
                --lr 2e-4 \
                --gpuid 0 \
                --early-stop 20 \
                --update-lr 10 \
                --epochloss_init 10000
```

To train the model by CRAG data, run the following command:

```
python main.py  --phase 'train' \
                --input-size 192 \
                --image-path './dataset/CRAG' \
                --model 'FCNsa' \
                --save-path 'weights/3/' \
                --alpha 0.0001 \
                --epoch 300 \
                --batchsize 4 \
                --n-channels 3 \
                --n-classes 2 \
                --lr 2e-4 \
                --gpuid 0 \
                --early-stop 20 \
                --update-lr 10 \
                --epochloss_init 10000
```
To train the model by GlaS data, run the following command:

```
python main.py  --phase 'train' \
                --input-size 192 \
                --image-path './dataset/GlaS' \
                --model 'FCNsa' \
                --save-path 'weights/4/' \
                --alpha 0.0001 \
                --epoch 300 \
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
To test the model by DRIVE data, run the following command:
```
python main.py  --phase 'test'
                --model-path './weights/1/CMD-Net_DRIVE.th' \
                --test-path './dataset/DRIVE/test/images/' \
                --gt-path './dataset/DRIVE/test/1st_manual' \
                --model 'FCNsa' \
                --n-channels 3 \
                --n-classes 1 \
                --gpuid 0

```  

To test the model by CHASEDB data, run the following command:
```
python main.py  --phase 'test' \
                --model-path './weights/2/CMD-Net_CHASEDB.th' \
                --test-path './dataset/CHASEDB1/test/images/' \
                --gt-path './dataset/CHASEDB1/test/labels1' \
                --model 'FCNsa' \
                --n-channels 3 \
                --n-classes 1 \
                --gpuid 0

```  
To test the model by CRAG data, run the following command:
```
python main.py  --phase 'test' \
                --model-path './weights/3/CMD-Net_CRAG.th' \
                --test-path './dataset/CRAG/test/img/' \
                --gt-path './dataset/CRAG/test/mask' \
                --model 'FCNsa' \
                --n-channels 3 \
                --n-classes 2 \
                --gpuid 0

```  
To test the model by GlaS data, run the following command:
```
python main.py  --phase 'test' \
                --model-path './weights/4/CMD-Net_GlaS.th' \
                --test-path './dataset/GlaS/test/img/' \
                --gt-path './dataset/GlaS/test/mask' \
                --model 'FCNsa' \
                --n-channels 3 \
                --n-classes 2 \
                --gpuid 0

```  
