import logging
import os
import sys
import tempfile
from glob import glob
from PIL import Image
import argparse
import yaml
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader
from monai.networks.layers.factories import Norm

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AddChanneld, AsDiscrete, Compose, LoadImaged, SaveImage, ScaleIntensityd, EnsureTyped, EnsureType
from DecorNet import unet

monai.config.print_config()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./runs/Apr25_15-24-31_hitlab-SYS-7048GR-TR')
parser.add_argument('--is_best', type=bool, default=False,
                        help='whether use best model in val')
parser.add_argument('--is_best_3D', type=bool, default=False,
                        help='whether use best model 3D in val')
parser.add_argument('--GPU_id', type=int, default=0)
config = parser.parse_args()
print(config)

# load config
with open(glob(os.path.join(config.model_path,'*.yaml'))[0], 'r') as f:
    hyper = yaml.full_load(f)

with open(os.path.join(config.model_path,'argparse.json')) as json_file:
    config_ = json.load(json_file)
print(config_)

image_path_val = hyper['DATA_CONFIG']['image_path_test']
seg_path_val = hyper['DATA_CONFIG']['seg_path_test']

# which model to load
if config.is_best:
    load_model =os.path.join(config.model_path,'best_metric_model_segmentation2d_array.pth')
    print('load best_metric_model_segmentation2d_array')
elif config.is_best_3D:
    load_model =os.path.join(config.model_path,'best_3Ddice_model_segmentation2d_array.pth')
    print('load best_3Ddice_model_segmentation2d_array.pth')
else:
    load_model =os.path.join(config.model_path,'latest_model_segmentation2d_array.pth')
    print('load latest_model_segmentation2d_array')
    
output_path = os.path.join(config.model_path,'output_test')
torch.cuda.set_device(config.GPU_id)

window_size = hyper['DATA_PREPROCESS']['window_size']

# load validation data
images = sorted(glob(os.path.join(image_path_val, "Lung_***_***.png")))
segs = sorted(glob(os.path.join(seg_path_val, "Lung_***_***.png")))
print(f'Length of images:{len(images)} Length of segs:{len(segs)}')
val_files = [{"img": img, "seg": seg} for img, seg in zip(images[:], segs[:])]

# build directory
if not os.path.exists(output_path):
    os.makedirs(output_path)

# define transforms for image and segmentation
val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
        EnsureTyped(keys=["img", "seg"]),
    ]
)
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
# sliding window inference need to input 1 image in every iteration
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

post_trans = eval(hyper['DATA_PREPROCESS']['post_trans'])
post_trans2 = Compose([EnsureType(), Activations(softmax=True)]) # not discrete


# %%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = unet.UNet_5layers(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=config_['channel_setting'],
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout = 0.1,
        norm = Norm.BATCH).to(device)

model.load_state_dict(torch.load(load_model,map_location=device))

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
model.eval()
count = 0
with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels, val_name = val_data["img"].to(device), val_data["seg"].to(device),val_data['img_meta_dict']['filename_or_obj'][0].split('/')[-1]

        # define sliding window size and batch size for windows inference
        roi_size = (window_size, window_size)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        val_outputs2 = [post_trans2(i) for i in decollate_batch(val_outputs)]
        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
        val_labels = decollate_batch(val_labels)
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)
        for val_output in val_outputs2:
            im = Image.fromarray((255*val_output[1,:,:].detach().cpu().numpy()).astype('uint8'))
            im.save(os.path.join(output_path,val_name))

    # aggregate the final mean dice result
    dice2D = dice_metric.aggregate().item()
    print("evaluation metric:",dice2D)
    # reset the status
    dice_metric.reset()

# record dice2D    
import json
save_path = os.path.join(config.model_path,'dice2D_test.json')
with open(save_path, "w") as outfile:
    json.dump({'2D dice':dice2D}, outfile)
# %%







