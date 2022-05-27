# %%
import numpy as np
import torch
from collections import OrderedDict

# %%
import torch.nn as nn
import torch.functional as F

# %%
from torch.utils.data import DataLoader
from torchvision.datasets.voc import VOCSegmentation
import ext_transforms as et

# %%
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, deeplabv3_resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP, ASPPConv, ASPPPooling

# %%
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transforms = et.ExtCompose([
             et.ExtResize(520),
             et.ExtCenterCrop(520),
             et.ExtToTensor(),
             et.ExtNormalize(mean, std)])
data = VOCSegmentation(root='/home/ubuntu/efs/VOC', transforms=transforms)
loader = DataLoader(data, batch_size=2, num_workers=0)

# %%
i = iter(loader)
data, lab = next(i)

# %%
net = resnet50(replace_stride_with_dilation=[False, True, True])

# %%
return_layers = {"layer4": "out"}
backbone = IntermediateLayerGetter(net, return_layers=return_layers)

# %%
clf = DeepLabHead(2048, 21)

# %%
input_shape = data.shape[-2:]

# %%
feat = backbone(data)


# %%
result = OrderedDict()
x = feat["out"]

# %%
x.shape


# %%
x = clf(x)

# %%
x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
result["out"] = x

# %%
a = ASPP(2048, [12, 24, 36])

# %%
_res = []

# %%
len(a.convs)

# %%
a.convs[4](x)

# %%
for conv in a.convs:
    _res.append(conv(x))

# %%
in_channels = 2048
out_channels = 256
b = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

# %%
c = nn.Sequential(nn.AdaptiveAvgPool2d(1)) #, nn.Conv2d(in_channels, out_channels, 1, bias=False)) #, nn.BatchNorm2d(out_channels))
# %%
c(x).shape
# %%
x.shape

# %%
test = deeplabv3_resnet50(num_classes=21)

# %%
test(data)

# %%
