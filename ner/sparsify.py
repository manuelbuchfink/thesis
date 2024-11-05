
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
import numpy as np
import wandb
import gc
import time
import h5py
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


from ner.FanBeam.ct_2d_projector import FanBeam2DProjector

from data import ImageDataset_2D_hdf5_canny, ImageDataset_2D_sparsify
from utils import get_config, get_sub_folder, get_data_loader_hdf5_canny, save_volume

from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from skimage.metrics import mean_squared_error  as mse # pylint: disable=import-error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import profile_line
import matplotlib.pyplot as plt
from utils import save_image_2d

sys.path.append('zhome/buchfiml/miniconda3/envs/odl/lib/python3.11/site-packages')
sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--start', type=str, default='.', help="starting slice")
parser.add_argument('--end', type=str, default='.', help="ending slice")
parser.add_argument('--id', type=str, default='.', help="id slice")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

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

# Setup data loader
print('Load volume: {}'.format(config['img_path']))
dataset = ImageDataset_2D_sparsify(config['img_path'], parser)
data_loader = get_data_loader_hdf5_canny(dataset, batch_size=config['batch_size'])

fbp_volume = [None] * 512
for it, (image) in enumerate(data_loader):

        # Input coordinates (h,w) grid and target image
        image = image.cuda()    # [1, h, w, 1], value range = [0, 1]

        ct_projector_sparse_view = FanBeam2DProjector(512, 512, proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])

        projections = ct_projector_sparse_view.forward_project(image.transpose(1, 3).squeeze(1))     # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
        fbp_recon= ct_projector_sparse_view.backward_project(projections)                          # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

        fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 3)                                          # [1, h, w, 1]
        torch.save(fbp_recon, os.path.join(image_directory, f"sparse_slice_{it + 1}.pt"))

sparse_images = []

for i in range(1, len(os.listdir(image_directory)) + 1):
    sparse_images.append(torch.load(os.path.join(image_directory, f'sparse_slice_{i}.pt')).cuda())

sparse_images = torch.cat(sparse_images, 0).cpu().detach().numpy()
save_volume(sparse_images, image_directory, config, "fbp_volume")