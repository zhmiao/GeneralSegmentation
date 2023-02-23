import os
import tarfile
from torchvision.datasets.utils import download_url, check_integrity
            
root = '/home/zhongqimiao/ssdprivate/datasets/VOC'
url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
filename = 'VOCtrainval_11-May-2012.tar'
md5 = '6cd6e144f989b92b3379bac3b3de84fd'

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

download_extract(url, root, filename, md5)
