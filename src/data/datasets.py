import os
import json
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .utils import register_dataset_obj 
from .ext_transforms import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': ExtCompose([
            ExtRandomScale((0.5, 2.0)),
            ExtRandomCrop(size=(513, 513), pad_if_needed=True),
            ExtRandomHorizontalFlip(),
            ExtToTensor(),
            ExtNormalize(mean, std),
        ]),
    'val': ExtCompose([
            ExtResize(513),
            ExtCenterCrop(513),
            ExtToTensor(),
            ExtNormalize(mean, std),
        ]),
}


class VOC(Dataset):
    def __init__(self,
                 rootdir,
                 dset='train',
                 transforms=None):

        self.transforms = transforms

        image_dir = os.path.join(rootdir, 'JPEGImages')

        if dset=='train':
            mask_dir = os.path.join(rootdir, 'SegmentationClassAug')
            split_f = os.path.join(rootdir, 'train_aug.txt')#'./datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(rootdir, 'SegmentationClass')
            splits_dir = os.path.join(rootdir, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, '{}.txt'.format(dset))

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, '{}.jpg'.format(x)) for x in file_names]
        self.masks = [os.path.join(mask_dir, '{}.png'.format(x)) for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


@register_dataset_obj('VOC')
class VOC_DM(pl.LightningDataModule):
    def __init__(self, conf):
        self.conf = conf
        self.prepare_data_per_node = True 
        self._log_hyperparams = False

        print("Loading data...")
        self.dset_tr = VOC(rootdir=self.conf.dataset_root,
                           dset='train',
                           transforms=data_transforms['train'])

        self.dset_te = VOC(rootdir=self.conf.dataset_root,
                           dset='val',
                           transforms=data_transforms['val'] )

        self.dset_te = VOC(rootdir=self.conf.dataset_root,
                           dset='val',
                           transforms=data_transforms['val'])

        print("Done.")

    def train_dataloader(self):
        return DataLoader(
            self.dset_tr, batch_size=self.conf.batch_size, shuffle=True, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=True, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dset_te, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=True, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.dset_te, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=True, persistent_workers=True
        )