
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
import warnings
import numpy as np
import time
import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error
import torch.nn.functional as F # pylint: disable=import-error
import h5py # pylint: disable=import-error


from ct_2d_projector import FanBeam2DProjector
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, get_image_pads
from data import ImageDataset_2D_hdf5


sys.path.append('zhome/buchfiml/miniconda3/envs/odl/lib/python3.11/site-packages')
sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

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
dataset = ImageDataset_2D_hdf5(config['img_path'], config['img_size'], config['num_slices'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

fbp_volume = []
for it, (grid, image, image_size) in enumerate(data_loader):

    # Input coordinates (h,w) grid and target image
    grid = grid.cuda()      # [1, h, w, 2], value range = [0, 1]
    image = image.cuda()    # [1, h, w, 1], value range = [0, 1]
    torch.save(image, os.path.join(image_directory, f"image_{it + 1}.pt"))
#     image_height = int(image_size[0][0] - image_size[1][0]) # 00 rmax, 01 rmin, 02 cmax, 03 cmin
#     image_width = int(image_size[2][0] - image_size[3][0])

#     pads = get_image_pads(image_size, config) # pads: rt, rb, cl,  cr

#     if image_height == 0 or image_width == 0: # skip emty images
#         skip_image = torch.zeros(1, 512, 512)
#         fbp_volume.append(skip_image.cuda())
#         continue

#     ct_projector_full_view = FanBeam2DProjector(image_height=image_height, image_width=image_width, proj_size=config['proj_size'], num_proj=config['num_proj_full_view'])
#     ct_projector_sparse_view = FanBeam2DProjector(image_height=image_height, image_width=image_width, proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])
#     projectors = [ct_projector_full_view, ct_projector_sparse_view]

#     projections = ct_projector_sparse_view.forward_project(image.transpose(1, 3).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
#     fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

#     fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 3)
#     fbp_padded = F.pad(fbp_recon, (0,0, pads[2],pads[3], pads[0],pads[1]))# [1, h, w, 1]
#     fbp_volume.append(fbp_padded.squeeze(3).cuda())

# # save corrected slices in new hdf5 Volume
# fbp_volume_path = os.path.join(image_directory, f"../{config['data'][:-3]}_fbp_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")
# print(f"saved to {config['data'][:-3]}_corrected_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")
# fbp_volume = torch.cat(fbp_volume, 0)
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
#     save_image(torch.tensor(slices_sparse[i], dtype=torch.float32), f"./u_volume_corrected_after_saving/image from saved volume, slice Nr. {i}.png")

