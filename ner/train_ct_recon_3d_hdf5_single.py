
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
import wandb
import numpy as np
import h5py # pylint: disable=import-error
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_image, save_image_2d, save_volume

from networks import Positional_Encoder_3D, FFN_3D
from ct_3d_projector import ConeBeam3DProjector

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error
import torch.nn.functional as F # pylint: disable=import-error
import tensorboardX # pylint: disable=import-error
from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from data import ImageDataset_3D_hdf5

warnings.filterwarnings("ignore")
sys.path.append('zhome/buchfiml/miniconda3/envs/odl/lib/python3.11/site-packages')
sys.path.append(os.getcwd())

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
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

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Setup data loader
print('Load volume: {}'.format(config['img_path']))
dataset = ImageDataset_3D_hdf5(config['img_path'], config['dataset_size'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])



for it, (grid, image) in enumerate(data_loader):

    # Input coordinates (h,w) grid and target image
    grid = grid.cuda()      # [1, h, w, d, 3], value range = [0, 1]
    image = image.cuda()    # [1, h, w, d, 1], value range = [0, 1]

    print(grid.shape, image.shape)


    '''
    downsample data
    '''
    image = torch.load(os.path.join(image_directory, f"fbp_volume.pt")).cuda() # [512, 512, 512]
    image = torch.tensor(image, dtype=torch.float16)[None, ...]  # [B, C, H, W]
    image = F.interpolate(image, size=(config['dataset_size'][1], config['dataset_size'][2]), mode='bilinear', align_corners=False)
    image = image.unsqueeze(4)

    ct_projector_sparse_view = ConeBeam3DProjector(config['img_size'], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])

    projections = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
    # fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

    # fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4)                                       # [1, h, w, 1]


    # Setup input encoder:
    encoder = Positional_Encoder_3D(config['encoder'])

    # Setup model
    model = FFN_3D(config['net'])


    model.cuda()
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    loss_fn = torch.nn.MSELoss().cuda()

    train_embedding = encoder.embedding(grid)  # fourier feature embedding:  ([1, x, y, z, 3] * [3, embedding_size]) -> [1, z, x, y, embedding_size]

    # Train model
    for iterations in range(max_iter):

        model.train()
        optim.zero_grad()
        with torch.cuda.amp.autocast():
            train_output = model(train_embedding)  # train model on grid: ([1, x, y, embedding_size]) > [1, x, y, 1]
            if iterations == max_iter - 1:
                model_train = train_output.squeeze().cuda().cpu().detach().numpy()
                save_volume(model_train, image_directory, config, "model_train")

            train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 4).squeeze(1)).to("cuda")      # evaluate by forward projecting
            train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))                                    # compare forward projected grid with sparse view projection

        train_loss.backward()
        optim.step()

        # Compute ssim
        if (iterations + 1) % config['val_iter'] == 0:

            model.eval()
            with torch.no_grad():
                fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 4)
                test_ssim = compare_ssim(fbp_prior.transpose(1,4).squeeze().cpu().numpy(), image.transpose(1,4).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)

            end = time.time()

            print("[Volume Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, (end - start) / 60))


            if (iterations + 1) == max_iter:
                torch.save(train_output, os.path.join(image_directory, f"prior_volume_{opts.id}.pt"))


# '''
# adapted from
# https://arxiv.org/pdf/2108.10991
# NeRP: Implicit Neural Representation Learning
# with Prior Embedding for Sparsely Sampled
# Image Reconstruction
# Liyue Shen, John Pauly, Lei Xing
# '''
# import sys
# import os
# import argparse
# import shutil
# import gc
# import time
# import warnings

# from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_volume

# from networks import Positional_Encoder_3D, FFN_3D
# from ct_3d_projector import ConeBeam3DProjector

# import torch # pylint: disable=import-error
# import torch.backends.cudnn as cudnn # pylint: disable=import-error
# import torch.nn.functional as F # pylint: disable=import-error
# import tensorboardX # pylint: disable=import-error
# from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
# from data import ImageDataset_3D_hdf5

# warnings.filterwarnings("ignore")
# sys.path.append('zhome/buchfiml/miniconda3/envs/odl/lib/python3.11/site-packages')
# sys.path.append(os.getcwd())

# start = time.time()

# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='', help='Path to the config file.')
# parser.add_argument('--output_path', type=str, default='.', help="outputs path")
# parser.add_argument('--id', type=str, default='.', help="id slice")
# # Load experiment setting
# opts = parser.parse_args()
# config = get_config(opts.config)
# max_iter = config['max_iter']

# torch.cuda.empty_cache()
# gc.collect()
# cudnn.benchmark = True

# # Setup output folder
# output_folder = os.path.splitext(os.path.basename(opts.config))[0]


# output_subfolder = config['data']
# model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}_{}' \
# .format(config['img_size'], config['num_proj_sparse_view'], config['model'], \
#     config['net']['network_input_size'], config['net']['network_width'], \
#     config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding'], config['description']))
# if not(config['encoder']['embedding'] == 'none'):
#     model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
# print(model_name)

# train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
# output_directory = os.path.join(opts.output_path + "/outputs", model_name)
# checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# # Setup data loader
# print('Load volume: {}'.format(config['img_path']))
# dataset = ImageDataset_3D_hdf5(config['img_path'], config['dataset_size'])
# data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

# for it, (grid, image) in enumerate(data_loader):

#     # Input coordinates (h,w) grid and target image
#     grid = grid.cuda()      # [1, h, w, d, 3], value range = [0, 1]
#     image = image.cuda()    # [1, h, w, d, 1], value range = [0, 1]

#     print(grid.shape, image.shape)

#     ct_projector_sparse_view = ConeBeam3DProjector(config['img_size'], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])

#     projections = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]

#     image = image.squeeze().cuda()
#     image = torch.tensor(image, dtype=torch.float16)[None, ...]  # [B, C, H, W]
#     image = F.interpolate(image, size=(config['dataset_size'][1], config['dataset_size'][2]), mode='bilinear', align_corners=False)
#     image = image.unsqueeze(4)

#     # Setup input encoder:
#     encoder = Positional_Encoder_3D(config['encoder'])

#     # Setup model
#     model = FFN_3D(config['net'])


#     model.cuda()
#     model.train()

#     optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
#     loss_fn = torch.nn.MSELoss().cuda()

#     train_embedding = encoder.embedding(grid)  # fourier feature embedding:  ([1, x, y, z, 3] * [3, embedding_size]) -> [1, z, x, y, embedding_size]

#     # Train model
#     for iterations in range(max_iter):

#         model.train()
#         optim.zero_grad()
#         with torch.cuda.amp.autocast():
#             train_output = model(train_embedding)  # train model on grid: ([1, x, y, embedding_size]) > [1, x, y, 1]

#             train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 4).squeeze(1)).to("cuda")      # evaluate by forward projecting
#             train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))                                    # compare forward projected grid with sparse view projection

#         train_loss.backward()
#         optim.step()

#         # Compute ssim
#         if (iterations + 1) % config['val_iter'] == 0:

#             model.eval()
#             with torch.no_grad():
#                 fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 4)
#                 test_ssim = compare_ssim(fbp_prior.transpose(1,4).squeeze().cpu().numpy(), image.transpose(1,4).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)

#             end = time.time()

#             print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, (end - start) / 60))


#             if (iterations + 1) == max_iter:
#                 prior_volume = train_output

# prior_volume = prior_volume.squeeze()
# prior_volume = torch.tensor(prior_volume, dtype=torch.float32)[None, ...]
# prior_volume = F.interpolate(torch.tensor(prior_volume, dtype=torch.float16), size=(512, 512), mode='bilinear', align_corners=False)
# prior_volume = prior_volume.unsqueeze(0)
# prior_volume = prior_volume.transpose(1, 4).squeeze(1)
# print(prior_volume.shape)


# ct_projector_full_view = ConeBeam3DProjector(config['fbp_img_size'], proj_size=config['proj_size'], num_proj=512)
# ct_projector_sparse_view = ConeBeam3DProjector(config['fbp_img_size'], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])



# projs_prior_full_view = ct_projector_full_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
# fbp_prior_full_view = ct_projector_full_view.backward_project(projs_prior_full_view)
# fbp_prior_full_view = fbp_prior_full_view.unsqueeze(1).transpose(1, 4)

# projs_prior_sparse_view = ct_projector_sparse_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
# fbp_prior_sparse_view = ct_projector_sparse_view.backward_project(projs_prior_sparse_view)
# fbp_prior_sparse_view = fbp_prior_sparse_view.unsqueeze(1).transpose(1, 4)

# streak_volume = (fbp_prior_sparse_view - fbp_prior_full_view)#.unsqueeze(1).transpose(1, 4)

# fbp_volume = torch.tensor(image, dtype=torch.float16)[None, ...]
# fbp_volume = F.interpolate(torch.tensor(fbp_volume, dtype=torch.float16), size=(512, 512), mode='bilinear', align_corners=False)

# corrected_volume = (fbp_volume.unsqueeze(4) - streak_volume).squeeze().cpu().detach().numpy()

# # fbp_prior_full_view = fbp_prior_full_view.squeeze().cpu().detach().numpy()
# # fbp_volume = fbp_volume.squeeze().cpu().detach().numpy()
# # prior_volume = prior_volume.squeeze().cuda().cpu().detach().numpy()
# # streak_volume = streak_volume.squeeze().cuda().cpu().detach().numpy()

# save_volume(corrected_volume, image_directory, config, "corrected_volume")
