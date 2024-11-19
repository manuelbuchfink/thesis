
'''
adapted from
https://arxiv.org/pdf/2108.10991
NeRP: Implicit Neural Representation Learning
with Prior Embedding for Sparsely Sampled
Image Reconstruction
Liyue Shen, John Pauly, Lei Xing
'''

import os
import argparse
import shutil
import gc
import time
import h5py
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from utils import get_config, get_data_loader_hdf5, save_volume, compute_vif, prepare_sub_folder

from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from skimage.metrics import mean_squared_error  as mse # pylint: disable=import-error
from skimage.metrics import peak_signal_noise_ratio as psnr


from utils import get_config, get_sub_folder, save_image, save_volume

warnings.filterwarnings("ignore")

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
prior_images = []
fbp_images = []
images = []
for i in range(1, len(os.listdir(image_directory)) / 4 + 1):
    corrected_images.append(torch.load(os.path.join(image_directory, f'corrected_slice_{i}.pt')).cuda())
    prior_images.append(torch.load(os.path.join(image_directory, f'prior_slice_{i}.pt')).cuda())
    fbp_images.append(torch.load(os.path.join(image_directory, f'fbp_slice_{i}.pt')).cuda())
    images.append(torch.load(os.path.join(image_directory, f'image_slice_{i}.pt')).cuda())


images = torch.cat(images, 0)
corrected_images = torch.cat(corrected_images, 0)
prior_images = torch.cat(prior_images, 0).squeeze().cpu().detach().numpy()
fbp_images = torch.cat(fbp_images, 0).squeeze().cpu().detach().numpy()


test_vif = compute_vif(images, corrected_images)

corrected_images = corrected_images.squeeze().cpu().detach().numpy()
images = images.squeeze().cpu().detach().numpy()
test_mse = mse(images, corrected_images)
test_ssim = compare_ssim(images, corrected_images, axis=-1, data_range=1.0)
test_psnr = psnr(images, corrected_images, data_range=1.0)

print(f"FINAL SSIM: {test_ssim}, MSE: {test_mse}, PSNR: {test_psnr}, VIF: {test_vif}")

# save_volume(fbp_images, image_directory, config, "fbp_volume")
# save_volume(corrected_images, image_directory, config, "corrected_volume")
# save_volume(prior_images, image_directory, config, "prior_volume")