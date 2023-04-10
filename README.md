# DECOR-Net
Code for paper: **DECOR-NET: A COVID-19 LUNG INFECTION SEGMENTATION NETWORK IMPROVED BY EMPHASIZING LOW-LEVEL FEATURES AND DECORRELATING FEATURES**. 
Specifically, we design a channel re-weighting strategy and a decorrelation loss to improve COVID-19 infection segmentation.
The model is built based on the MONAI framework.

## Usage
### Installation
1. Requirements

- Linux
- Python 3.6+

2. Dependencies.
- numpy>=1.21.5
- pandas>=1.1.5
- torch>=1.7.0
- torchvision>=0.11.1
- monai>=0.8.1
- pillow
- yaml
- json
- torchmetrics


### Dataset
The preprocessed COVID-19 Challenge dataset can be found [BaiduDisk](https://pan.baidu.com/s/1fWKTKGIkhsgnbGKx3EdPSQ) (key = tu1h).

The COVID-19 Challenge dataset can be found [here](https://covid-segmentation.grand-challenge.org).


### Training and Evaluation
The path of dataset need to be set in **./CTE_Net/CTE-Net.yaml** before training.
```
python train.py
```

### Evaluation
```
bash evaluation.sh
```
The path of `training_dir` in **evaluation.sh** need to be set before evaluation, so the code knows which model needs to be evaluated. The folder for `training_dir` will be automatically generated by  **train.py**.
All the metrics generated by **evaluation.sh** will be save in the folder of `training_dir`.
