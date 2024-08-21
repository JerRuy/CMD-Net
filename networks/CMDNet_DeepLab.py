import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from networks.utils.CMDNet_DeepLab_res import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from networks.utils.CMDNet_DeepLab_utils import ASPP, ASPP_Bottleneck


class CMDNet_DeepLab(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super(CMDNet_DeepLab, self).__init__()

        self.num_classes = n_classes
        self.resnet = ResNet18_OS8() 
        self.aspp = ASPP(num_classes=self.num_classes) 

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) 

        # output = self.aspp(feature_map)
        # output = F.upsample(output, size=(h, w), mode="bilinear")

        down = self.aspp(feature_map)
        output = F.upsample(down, size=(h, w), mode="bilinear")

        output = torch.sigmoid(output)
        return output, down
