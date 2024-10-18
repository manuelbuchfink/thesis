
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
from utils import get_config, get_sub_folder, get_data_loader_hdf5_canny, reshape_tensor, correct_image_slice, get_image_pads, reshape_model_weights, save_image
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

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = get_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Setup data loader
print('Load volume: {}'.format(config['img_path']))
dataset = ImageDataset_2D_hdf5(config['img_path'], config['img_size'], config['num_slices'], parser)
data_loader = get_data_loader_hdf5_canny(dataset, batch_size=config['batch_size'])


pretrain = False
skip = False
skips = 0
previous_projection = None
total_its = 0
zeros = 0
avg_ssim_recon = 0
avg_ssim_train = 0
print(f"start = {opts.start}, end = {opts.end}")
for it, (grid, image, image_size) in enumerate(data_loader):
    if it >= int(opts.start) and it < int(opts.end):
        # Input coordinates (h,w) grid and target image
        grid = grid.cuda()      # [1, h, w, 2], value range = [0, 1]
        image = image.cuda()    # [1, h, w, 1], value range = [0, 1]
        '''
        compute image height and image width with canny edge pipeline
        '''

        # print(f"canny shape {canny_volume.shape}")
        image_height = int(image_size[0][0] - image_size[1][0]) # 00 rmax, 01 rmin, 02 cmax, 03 cmin
        image_width = int(image_size[2][0] - image_size[3][0])

        pads = get_image_pads(image_size, config) # pads: rt, rb, cl,  cr

        if image_height == 0 or image_width == 0: # skip emty images
            skip_image = torch.zeros(1, 512, 512, 1).to(torch.float16)
            torch.save(skip_image, os.path.join(image_directory, f"corrected_slice_{it + 1}.pt"))
            zeros+=1
            continue

        ct_projector_full_view = FanBeam2DProjector(image_height=image_height, image_width=image_width, proj_size=config['proj_size'], num_proj=config['num_proj_full_view'])
        ct_projector_sparse_view = FanBeam2DProjector(image_height=image_height, image_width=image_width, proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])
        projectors = [ct_projector_full_view, ct_projector_sparse_view]

        projections = ct_projector_sparse_view.forward_project(image.transpose(1, 3).squeeze(1)).to(torch.float16)     # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
        fbp_recon= ct_projector_sparse_view.backward_project(projections).to(torch.float16)                            # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

        train_projections = projections[..., np.newaxis].to(torch.float16)                                             # [1, num_proj_sparse_view, original_image_size, 1]
        fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 3).to(torch.float16)                                           # [1, h, w, 1]

        # Setup input encoder:
        encoder = Positional_Encoder(config['encoder'], bb_embedding_size= int(image_height + image_width))

        # Setup model
        model = FFN(config['net'], int(image_height + image_width))

        '''
        Check if sequential slices are similar enough in order to skip training for one step and reuse the previous prior to compute the corrected image
        '''
        if pretrain:
            fbp_prev = ct_projector_sparse_view.backward_project(previous_projection).unsqueeze(1).transpose(1, 3).to(torch.float16)
            sequential_ssim = compare_ssim(fbp_prev.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
            print(f"Sequential SSIM = {sequential_ssim}")

            if sequential_ssim > config['slice_skip_threshold']:
                print(f"SSIM passed, skipping training for slice nr. {it + 1}")
                skip = True
                skips+=1
                previous_image, ssim_recon, ssim_train = correct_image_slice(skip, zeros, train_output, projectors, image, fbp_recon, train_projections, pads, it, iterations, image_directory, config)
                avg_ssim_recon += ssim_recon
                avg_ssim_train += ssim_train
                continue
            else:
                skip = False

            # Load pretrain model
            state_dict = reshape_model_weights(image_height, image_width, config, checkpoint_directory, opts.id)

            model.load_state_dict(state_dict['net'])


        model.cuda()
        model.train()

        optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
        loss_fn = torch.nn.MSELoss().cuda()

        train_embedding = encoder.embedding(grid).to(torch.float16)   # fourier feature embedding:  ([1, x, y, 2] * [2, embedding_size]) -> [1, x, y, embedding_size]

        # Train model
        for iterations in range(max_iter):

            model.train()
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                train_output = model(train_embedding)  # train model on grid: ([1, x, y, embedding_size]) > [1, x, y, 1]

                train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 3).squeeze(1)).to("cuda")      # evaluate by forward projecting
                train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))                                    # compare forward projected grid with sparse view projection

            train_loss.backward()
            optim.step()

            # Compute ssim
            if (iterations + 1) % config['val_iter'] == 0:
                if config['eval']:
                    model.eval()
                    with torch.no_grad():
                        test_loss = 0.5 * loss_fn(train_output.to("cuda"), image.to("cuda")) # compare grid with test image
                        test_psnr = - 10 * torch.log10(2 * test_loss).item()
                        test_loss = test_loss.item()

                        test_ssim = compare_ssim(train_output.transpose(1,3).squeeze().cpu().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)

                    end = time.time()

                    train_writer.add_scalar('test_loss', test_loss, iterations + 1)
                    train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)

                    print("[Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g} | Time Elapsed {}".format(iterations + 1, max_iter, test_loss, test_psnr, test_ssim, (end - start)))
                else:
                    model.eval()
                    with torch.no_grad():
                        fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 3)
                        test_ssim = compare_ssim(fbp_prior.transpose(1,3).squeeze().cpu().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)

                    end = time.time()

                    print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, (end - start) / 60))

                # get prior image once training is finished
                if test_ssim > config['accuracy_goal']: # stop early if accuracy is above threshold
                    previous_projection, ssim_recon, ssim_train  = correct_image_slice(skip, zeros, train_output, projectors, image, fbp_recon, train_projections, pads, it, iterations, image_directory, config)
                    avg_ssim_recon += ssim_recon
                    avg_ssim_train += ssim_train
                    break

                if (iterations + 1) == max_iter:
                    previous_projection, ssim_recon, ssim_train  = correct_image_slice(skip, zeros, train_output, projectors, image, fbp_recon, train_projections, pads, it, iterations, image_directory, config)
                    avg_ssim_recon += ssim_recon
                    avg_ssim_train += ssim_train

            total_its+=1


        print(f"Nr. of skips: {skips}")
       # print(f"avg ssim TRAIN: {avg_ssim_train /512}, avg ssim RECON: {avg_ssim_recon /512}")
        # Save current model
        model_name = os.path.join(checkpoint_directory, f'temp_model_{opts.id}.pt')
        torch.save({'net': model.state_dict(), \
                    'enc': encoder.B, \
                    'opt': optim.state_dict(), \
                    }, model_name)

        pretrain = True
