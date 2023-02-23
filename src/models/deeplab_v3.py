import os
import copy
from collections import OrderedDict
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP
from torchvision.models._utils import IntermediateLayerGetter


__all__ = [
    'DeepLabV3_ResNet50'
]

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabV3_ResNet50(nn.Module):

    name = 'DeepLabV3_ResNet50'

    def __init__(self, num_cls=21, output_stride=8):
        super(DeepLabV3_ResNet50, self).__init__()
        self.num_cls = num_cls
        self.feature = None
        self.classifier = None
        self.criterion_seg = None
        self.output_stride = output_stride

        # Model setup and weights initialization
        self.setup_net()

    def setup_net(self):
        if self.output_stride==8:
            replace_stride_with_dilation=[False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation=[False, False, True]
            aspp_dilate = [6, 12, 18]

        self.feature = IntermediateLayerGetter(resnet50(pretrained=True, 
                                                        replace_stride_with_dilation=replace_stride_with_dilation),
                                               return_layers = {'layer4': 'out'}) 
        self.classifier = DeepLabHead(2048, self.num_cls, aspp_dilate)

        for m in self.feature.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01

    def setup_criteria(self):
        self.criterion_seg = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def forward(self, data):
        input_shape = data.shape[-2:]
        features = self.feature(data)
        outputs = self.classifier(features['out'])
        outputs = F.interpolate(outputs, size=input_shape, mode='bilinear', align_corners=False)
        return outputs

