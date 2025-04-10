
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


from ct_2d_projector import FanBeam2DProjector
from networks import Positional_Encoder, FFN
from data import ImageDataset_2D_hdf5_canny
from utils import get_config, get_sub_folder, get_data_loader_hdf5, reshape_tensor, correct_image_slice_all, get_image_pads, reshape_model_weights, save_image

from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from skimage.metrics import mean_squared_error  as mse # pylint: disable=import-error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import profile_line
import matplotlib.pyplot as plt


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
dataset = ImageDataset_2D_hdf5_canny(config['img_path'], parser, image_directory)
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])


pretrain = False
previous_projection = None
skip = False
skips = 0
total_its = 0
serial_skips = 0
trained_slices = 0
max_series = 0
zeros = 0
print(f"start = {opts.start}, end = {opts.end}")
for it, (grid, image, image_size) in enumerate(data_loader):
    if it >= int(opts.start) and it < int(opts.end):
        # Input coordinates (h,w) grid and target image
        grid = grid.cuda()      # [1, h, w, 2], value range = [0, 1]
        image = image.cuda()    # [1, h, w, 1], value range = [0, 1]
        '''
        compute image height and image width with canny edge pipeline
        '''

        image_height = int(image_size[0][0] - image_size[1][0]) # 00 rmax, 01 rmin, 02 cmax, 03 cmin
        image_width = int(image_size[2][0] - image_size[3][0])

        pads = get_image_pads(image_size, config) # pads: rt, rb, cl,  cr

        if image_height == 0 or image_width == 0: # skip emty images
            skip_image = torch.zeros(1, 512, 512, 1).to(torch.float16)
            torch.save(skip_image, os.path.join(image_directory, f"corrected_slice_{it + 1}.pt"))
            torch.save(skip_image, os.path.join(image_directory, f"prior_slice_{it + 1}.pt"))

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
                serial_skips+=1
                total_its+=1
                corrected_image, prior, train_projections = correct_image_slice_all(skip, train_output, projectors, image, fbp_recon, train_projections, pads, it, image_directory, config)

                continue
            else:
                max_series = np.maximum(max_series, serial_skips)
                trained_slices+=1
                serial_skips=0
                skip = False
            # Load pretrain model
            state_dict = reshape_model_weights(image_height, image_width, config, checkpoint_directory, opts.id)

            model.load_state_dict(state_dict['net'])


        model.cuda()
        model.train()

        optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
        loss_fn = torch.nn.MSELoss().cuda()

        train_embedding = encoder.embedding(grid).to(torch.float16)   # fourier feature embedding:  ([1, x, y, 2] * [2, embedding_size]) -> [1, x, y, embedding_size]
        scaler = torch.cuda.amp.GradScaler()
        # Train model
        for iterations in range(max_iter):

            model.train()
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                train_output = model(train_embedding)  # train model on grid: ([1, x, y, embedding_size]) > [1, x, y, 1]

                train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 3).squeeze(1)).to("cuda")      # evaluate by forward projecting
                train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))                                    # compare forward projected grid with sparse view projection
                #train_loss = (0.5 * loss_fn(train_output.to("cuda"), fbp_recon.to("cuda")))     # compare forward projected grid with sparse view projection

            scaler.scale(train_projections)
            scaler.scale(train_loss).backward()
            scaler.step(optim)
            scaler.update()

            # Compute ssim
            if (iterations + 1) % config['val_iter'] == 0:
                if config['eval']:
                    model.eval()
                    with torch.no_grad():
                        test_loss = 0.5 * loss_fn(train_output.to("cuda"), image.to("cuda")) # compare grid with test image
                        test_psnr = - 10 * torch.log10(2 * test_loss).item()
                        test_loss = test_loss.item()
                        fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 3)
                        test_ssim = compare_ssim(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), multichannel=True, data_range=1.0)
                        test_mse = mse(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy())
                        test_psnr = psnr(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), data_range=1.0)

                    end = time.time()

                    print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | MSE {:.4g} | PSNR {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, test_mse, test_psnr, (end - start) / 60))

                else:
                    model.eval()
                    with torch.no_grad():
                        fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 3)
                        fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 3)
                        test_ssim = compare_ssim(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), multichannel=True, data_range=1.0)
                        test_mse = mse(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy())
                        test_psnr = psnr(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), data_range=1.0)

                    end = time.time()

                    print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | MSE {:.4g} | PSNR {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, test_mse, test_psnr, (end - start) / 60))

                # get prior image once training is finished
                if test_ssim > config['accuracy_goal']: # stop early if accuracy is above threshold
                    corrected_image, prior, train_projections  = correct_image_slice_all(skip, train_output, projectors, image, fbp_recon, train_projections, pads, it, image_directory, config)

                    break

                if (iterations + 1) == max_iter:
                    corrected_image, prior, train_projections  = correct_image_slice_all(skip, train_output, projectors, image, fbp_recon, train_projections, pads, it, image_directory, config)


            total_its+=1


        print(f"Nr. of skips_{opts.id}: {skips}")
        print(f"Nr. of zeros_{opts.id}: {zeros}")
        print(f"Nr. of total_iterations_{opts.id}: {total_its}")
        print(f"Time Elapsed: {(end- start) / 60}")
       # print(f"avg ssim TRAIN: {avg_ssim_train /512}, avg ssim RECON: {avg_ssim_recon /512}")
        # Save current model
        model_name = os.path.join(checkpoint_directory, f'temp_model_{opts.id}.pt')
        torch.save({'net': model.state_dict(), \
                    'enc': encoder.B, \
                    'opt': optim.state_dict(), \
                    }, model_name)

        pretrain = True
        previous_projection = train_projections
        total_its+=1
