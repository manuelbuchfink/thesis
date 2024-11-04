'''
adapted from
https://arxiv.org/pdf/2108.10991
NeRP: Implicit Neural Representation Learning
with Prior Embedding for Sparsely Sampled
Image Reconstruction
Liyue Shen, John Pauly, Lei Xing
'''
import time
import sys
import os
import argparse
import shutil
import gc
import warnings

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error

from ct_3d_projector import ConeBeam3DProjector
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5
from data import ImageDataset_3D_hdf5

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
max_iter = config['max_iter']

torch.cuda.empty_cache()
gc.collect()
cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
print(f"output folder{output_folder}")

output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}_{}' \
.format(config['img_size'], config['num_proj_sparse_view'], config['model'], \
    config['net']['network_input_size'], config['net']['network_width'], \
    config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding'], config['description']))
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)


output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Setup data loader
print('Load volume: {}'.format(config['img_path']))
dataset = ImageDataset_3D_hdf5(config['img_path'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

for it, (grid, image) in enumerate(data_loader):

    n = config['down_sample_factor']
    image = image.squeeze()[::n,::n,::n].unsqueeze(0).unsqueeze(4).cuda()     # [1, h, w, d, 1], value range = [0, 1]

    ct_projector_sparse_view = ConeBeam3DProjector(image.squeeze().shape, num_proj=config['num_proj_sparse_view'])

    projections = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
    fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

    fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4)                                          # [1, h, w, 1]
    fbp_recon = torch.tensor(fbp_recon, dtype=torch.float16)

fbp_volume = fbp_recon.squeeze().cuda()
torch.save(fbp_volume, os.path.join(image_directory, f"fbp_volume.pt"))
torch.save(image.squeeze().cuda(), os.path.join(image_directory, f"image.pt"))
