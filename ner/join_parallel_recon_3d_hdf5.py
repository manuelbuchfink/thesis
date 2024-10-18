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
import gc
import time
import warnings
import numpy as np
import h5py # pylint: disable=import-error

from utils import get_config, get_sub_folder, save_volume
from ct_3d_projector import ConeBeam3DProjector

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error
import torch.nn.functional as F # pylint: disable=import-error

sys.path.append('zhome/buchfiml/miniconda3/envs/odl/lib/python3.11/site-packages')
sys.path.append(os.getcwd())
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
priors = []
for i in range(4):

    prior_volume = torch.load(os.path.join(image_directory, f'prior_volume_{i}.pt'))
    prior_volume = prior_volume.squeeze()
    prior_volume = torch.tensor(prior_volume, dtype=torch.float32)[None, ...]
    prior_volume = F.interpolate(torch.tensor(prior_volume, dtype=torch.float32), size=(512, 512), mode='bilinear', align_corners=False)
    priors.append(prior_volume.squeeze(0))

prior_volume  = torch.cat(priors, 0).unsqueeze(0)

fbp_volume = torch.load(os.path.join(image_directory, f'fbp_volume.pt'))
fbp_volume = fbp_volume.squeeze()
fbp_volume = torch.tensor(fbp_volume, dtype=torch.float32)[None, ...]

ct_projector_full_view = ConeBeam3DProjector(config['fbp_img_size'], proj_size=config['proj_size'], num_proj=512)
ct_projector_sparse_view = ConeBeam3DProjector(config['fbp_img_size'], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])

prior_volume = prior_volume.unsqueeze(4)

projs_prior_full_view = ct_projector_full_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
fbp_prior_full_view = ct_projector_full_view.backward_project(projs_prior_full_view)

projs_prior_sparse_view = ct_projector_sparse_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
fbp_prior_sparse_view = ct_projector_sparse_view.backward_project(projs_prior_sparse_view)

streak_volume = (fbp_prior_sparse_view - fbp_prior_full_view).unsqueeze(1).transpose(1, 4)

corrected_volume = (fbp_volume.unsqueeze(4) - streak_volume).squeeze().cpu().detach().numpy()

fbp_prior_full_view = fbp_prior_full_view.squeeze().cpu().detach().numpy()
fbp_volume = fbp_volume.squeeze().cpu().detach().numpy()
prior_volume = prior_volume.squeeze().cuda().cpu().detach().numpy()
streak_volume = streak_volume.squeeze().cuda().cpu().detach().numpy()

#save_volume(fbp_prior_full_view, image_directory, config, "fbp_prior_full_view")
save_volume(fbp_volume, image_directory, config, "fbp_volume")
save_volume(corrected_volume, image_directory, config, "corrected_volume")
save_volume(prior_volume, image_directory, config, "prior_volume")
save_volume(streak_volume, image_directory, config, "streak_volume")