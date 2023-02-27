# GeneralSegmentation

## Test data download
1) Open `supp/data_download.py`, change the root directory where you want to save the data to.
2) Run this `data_download.py` file to download the data. 

## Training and evaluation
1) Before training, change the dataset root in the configuration files in `config` folder.
2) Train: 
```
python main.py --config ./configs/voc_plain_051522.yaml --gpus 0,1,2,3 --logger_type comet --session 0 
```
Once the model is trained, a weight file will be saved to `weights` folder.
3) Evaluate:
```
python main.py --config ./configs/voc_plain_051522.yaml --gpus 0 --evaluate path_to_your_weights_file
```
**4) NOTE: Logit adjustment method only works for binary masks now. Don't use it on VOC.**


## Basic packages:
- pytorch
- torchvision
- pytorch-lightning
- numpy
- typer
- munch