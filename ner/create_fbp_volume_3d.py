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
import numpy as np

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error
import h5py # pylint: disable=import-error

from ct_3d_projector import ConeBeam3DProjector
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_image
from data import ImageDataset_3D_hdf5_direct


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
dataset = ImageDataset_3D_hdf5_direct(config['img_path'], config['fbp_img_size'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

for it, (grid, image) in enumerate(data_loader):

    image = image.cuda()    # [1, h, w, d, 1], value range = [0, 1]

    ct_projector_sparse_view = ConeBeam3DProjector(config['fbp_img_size'], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])
    #ct_projector_full_view = ConeBeam3DProjector(config['fbp_img_size'], proj_size=config['proj_size'], num_proj=512)

    projections = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
    fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

    fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4)                                          # [1, h, w, 1]
    # full_proj = ct_projector_full_view.forward_project(fbp_recon.transpose(1, 4).squeeze(1))
    # full_fbp = ct_projector_full_view.backward_project(full_proj)
    # full_fbp = full_fbp.unsqueeze(1).transpose(1, 4)

# save corrected slices in new hdf5 Volume
fbp_volume_path = os.path.join(image_directory, f"../{config['data'][:-3]}_fbp_with_{config['num_proj_sparse_view']}_projections.hdf5")
print(f"saved to {config['data'][:-3]}_fbp_with_{config['num_proj_sparse_view']}_projections.hdf5")

fbp_volume = fbp_recon.squeeze().cuda()
torch.save(fbp_volume, os.path.join(image_directory, f"fbp_volume.pt"))

# fbp_volume = fbp_volume.cpu().detach().numpy()

# gridSpacing=[5.742e-05, 5.742e-05, 5.742e-05]
# gridOrigin=[0, 0 ,0]
# with h5py.File(fbp_volume_path,'w') as hdf5:
#     hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
#     hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
#     hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
#     hdf5.create_dataset("Volume", data=np.asarray(fbp_volume))

# fbp_volume = h5py.File(fbp_volume_path, 'r')
# fbp_volume = fbp_volume['Volume']
# slices_sparse = [None] * 512

# for i in range(512):

#     #split image into N evenly sized chunks
#     slices_sparse[i] = fbp_volume[i,:,:].squeeze()           # (512,512) = [h, w]
#     save_image(torch.tensor(slices_sparse[i], dtype=torch.float32), f"./u_im_spare_after_saving/image from saved volume, slice Nr. {i}.png")



# '''
# full volume test direkt
# '''
# full_volume = full_fbp.squeeze().cuda()
# full_volume = full_volume.cpu().detach().numpy()

# # save corrected slices in new hdf5 Volume
# fbp_full_path = os.path.join(image_directory, f"../{config['data'][:-3]}_fbp_full_{config['num_proj_sparse_view']}_projections.hdf5")
# print(f"saved to {config['data'][:-3]}_fbp_full_{config['num_proj_sparse_view']}_projections.hdf5")



# with h5py.File(fbp_full_path,'w') as hdf5:
#     hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
#     hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
#     hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
#     hdf5.create_dataset("Volume", data=np.asarray(full_volume))

# full_volume = h5py.File(fbp_full_path, 'r')
# full_volume = full_volume['Volume']
# slices_sparse = [None] * 512

# for i in range(512):

#     #split image into N evenly sized chunks
#     slices_sparse[i] = full_volume[i,:,:].squeeze()           # (512,512) = [h, w]
#     save_image(torch.tensor(slices_sparse[i], dtype=torch.float32), f"./u_im_correct_after_saving/image from saved volume, slice Nr. {i}.png")
