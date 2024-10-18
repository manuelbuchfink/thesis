
'''
adapted from
https://arxiv.org/pdf/2108.10991
NeRP: Implicit Neural Representation Learning
with Prior Embedding for Sparsely Sampled
Image Reconstruction
Liyue Shen, John Pauly, Lei Xing
'''
import sys
import os
import argparse
import shutil
sys.path.append('zhome/buchfiml/miniconda3/envs/odl/lib/python3.11/site-packages')
sys.path.append(os.getcwd())
import wandb
import torch
import torch.backends.cudnn as cudnn
import tensorboardX
import numpy as np
import torch.nn.functional as F

from ct_2d_projector import FanBeam2DProjector
from data import ImageDataset_2D_hdf5
from networks import Positional_Encoder, FFN
from utils import get_config, get_sub_folder, save_image
from skimage.metrics import structural_similarity as compare_ssim

import gc
from datetime import datetime
import h5py
import warnings
warnings.filterwarnings("ignore")
from utils import save_image_2d
import time
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")


# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)

torch.cuda.empty_cache()
gc.collect()
cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]

output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}_{}' \
.format(config['img_size'], config['num_proj_sparse_view'], config['model'], \
    config['net']['network_input_size'], config['net']['network_width'], \
    config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding'], config['description']))
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = get_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

LOAD IMAGE SLICES INTO CORRECTED_IMAGES AND SPARSE_IMAGES FROM FILE


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
corrected_images = []

for i in range(1, len(os.listdir(image_directory)) + 1):
    corrected_images.append(torch.load(os.path.join(image_directory, f'corrected_slice_{i}.pt')).cuda())

corrected_images = torch.cat(corrected_images, 0).cpu().detach().numpy()
#sparse_images = torch.cat(sparse_images, 0).squeeze()
#print(f"total iterations: {total_its}")

# save corrected slices in new hdf5 Volume
corrected_image_path = os.path.join(image_directory, f"../{config['data'][:-3]}_corrected_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")
print(f"saved to {config['data'][:-3]}_corrected_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")

sparse_image_path = os.path.join(image_directory, f"../{config['data'][:-3]}_sparse_view_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")
print(f"saved to {config['data'][:-3]}_sparse_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")

gridSpacing=[5.742e-05, 5.742e-05, 5.742e-05]
gridOrigin=[0, 0 ,0]
with h5py.File(corrected_image_path,'w') as hdf5:
    hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
    hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
    hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
    hdf5.create_dataset("Volume", data=np.asarray(corrected_images))

# with h5py.File(sparse_image_path,'w') as hdf5:
#     hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
#     hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
#     hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
#     hdf5.create_dataset("Volume", data=np.asarray(sparse_images))

# image_fbp_direct = h5py.File(sparse_image_path, 'r')
# image_fbp_direct = image_fbp_direct['Volume']

# slices_sparse = [None] * (512)
# for i in range(512):

#     #split image into N evenly sized chunks
#     slices_sparse[i] = image_fbp_direct[i,:,:].squeeze()           # (512,512) = [h, w]
#     save_image(torch.tensor(slices_sparse[i], dtype=torch.float32), f"./u_im_spare_after_saving/image from saved volume, slice Nr. {i}.png")


image_correct = h5py.File(corrected_image_path, 'r')
image_correct = image_correct['Volume']
slices_correct = [None] * (512)
for i in range(512):

    #split image into N evenly sized chunks
    slices_correct[i] = image_correct[i,:,:].squeeze()           # (512,512) = [h, w]
    save_image(torch.tensor(slices_correct[i], dtype=torch.float32), f"./u_im_correct_after_saving/image from saved volume, slice Nr. {i}.png")

