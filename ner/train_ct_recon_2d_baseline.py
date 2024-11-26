
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
import time
import torch
import torch.backends.cudnn as cudnn
from vif_utils import vif
from ct_2d_projector_baseline import FanBeam2DProjector

import numpy as np

from networks import Positional_Encoder_dicom , FFN_base
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_image_2d
from data import ImageDataset_2D_dicom
from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from skimage.metrics import mean_squared_error  as mse # pylint: disable=import-error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import profile_line
import matplotlib.pyplot as plt
import gc
from datetime import datetime
import warnings
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


output_subfolder = config['data']
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


# Setup input encoder:
encoder = Positional_Encoder_dicom(config['encoder'])

# Setup model
model = FFN_base(config['net'])
model.cuda()
model.train()

optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
loss_fn = torch.nn.MSELoss().to("cuda")


# Setup data loader
print('Load image: {}'.format(config['img_path']))
dataset = ImageDataset_2D_dicom(config['img_path'],config['img_size'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])



for it, (grid, image) in enumerate(data_loader):
    if it > 336 and it < 340:
        # Input coordinates (x,y) grid and target image
        grid = grid.cuda()      # [1, x, y, 2], [0, 1]
        image = image.cuda()    # [1, x, y, 1], [0, 1]

        ct_projector_full_view_512 = FanBeam2DProjector(image.squeeze().shape[1], image_width=image.squeeze().shape[0], proj_size=config['proj_size'], num_proj=config['num_proj_full_view'])
        ct_projector_sparse_view = FanBeam2DProjector(image.squeeze().shape[1], image_width=image.squeeze().shape[0], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])

        projections = ct_projector_sparse_view.forward_project(image.transpose(1, 3).squeeze(1))  # ([1, y, x])        -> [1, num_proj, x]
        fbp_recon = ct_projector_sparse_view.backward_project(projections)                    # ([1, num_proj, x]) -> [1, y, x]

        train_projections = projections[..., np.newaxis]                  # [1, num_proj, x, 1]
        fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 3)  # [1, x, y, 1]

        save_image_2d(image.float(), os.path.join(image_directory, f"image_{it}.png"))
        save_image_2d(train_projections.float(), os.path.join(image_directory, "train128.png"))
        save_image_2d(fbp_recon.float(), os.path.join(image_directory, "fbprecon_128.png"))

        train_embedding = encoder.embedding(grid)  #  fourier feature embedding:  ([1, x, y, 2] * [2, embedding_size]) -> [1, x, y, embedding_size]

        # Train model
        for iterations in range(max_iter):

            model.train()
            optim.zero_grad()

            train_output = model(train_embedding)      #  train model on grid:        ([1, x, y, embedding_size])          -> [1, x, y, 1]

            train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 3).squeeze(1)).to("cuda")      # evaluate by forward projecting
            train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))                                          # compare forward projected grid with sparse view projection

            train_loss.backward()
            optim.step()

            # Compute training psnr
            if (iterations + 1) % config['log_iter'] == 0:
                train_psnr = -10 * torch.log10(2 * train_loss).item()
                train_loss = train_loss.item()
                end = time.time()

                fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 3)
                test_ssim = compare_ssim(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), multichannel=True, data_range=1.0)
                test_mse = mse(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy())
                test_psnr = psnr(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), data_range=1.0)

                print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | MSE {:.4g} | PSNR {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, test_mse, test_psnr, (end - start) / 60))

            # Compute testing psnr
            if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
                model.eval()
                with torch.no_grad():
                    test_output = model(train_embedding)      # train model on grid
                    test_loss = 0.5 * loss_fn(test_output.to("cuda"), image.to("cuda")) # compare grid with test image
                    test_psnr = - 10 * torch.log10(2 * test_loss).item()
                    test_loss = test_loss.item()
                    fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 3)
                    test_ssim = compare_ssim(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), multichannel=True, data_range=1.0)
                    test_mse = mse(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy())
                    test_psnr = psnr(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), data_range=1.0)

                end = time.time()
                save_image_2d(test_output, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(iterations + 1, test_psnr, test_ssim)))
                print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | MSE {:.4g} | PSNR {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, test_mse, test_psnr, (end - start) / 60))

        # Save final model
        model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
        torch.save({'net': model.state_dict(), \
                    'enc': encoder.B, \
                    'opt': optim.state_dict(), \
                    }, model_name)

        # get prior image once training is finished
        prior = test_output
        prior_train = train_output
        save_image_2d(prior, os.path.join(image_directory, "prior.png"))
        save_image_2d(prior_train, os.path.join(image_directory, "prior_train.png"))
        prior_diff = prior - prior_train
        save_image_2d(prior_diff, os.path.join(image_directory, "prior_diff.png"))

        projs_prior_512 = ct_projector_full_view_512.forward_project(prior.transpose(1, 3).squeeze(1))
        fbp_prior_512 = ct_projector_full_view_512.backward_project(projs_prior_512)

        projs_prior_128 = ct_projector_sparse_view.forward_project(prior.transpose(1, 3).squeeze(1))
        fbp_prior_128 = ct_projector_sparse_view.backward_project(projs_prior_128)


        streak_prior_128 = fbp_prior_128 - fbp_prior_512

        fbp_prior_512 = fbp_prior_512.unsqueeze(1).transpose(1, 3)
        fbp_prior_128 = fbp_prior_128.unsqueeze(1).transpose(1, 3)

        save_image_2d(fbp_prior_512, os.path.join(image_directory, "fbp_prior_512.png"))
        save_image_2d(fbp_prior_128, os.path.join(image_directory, "fbp_prior_128.png"))



        streak_prior_128 = streak_prior_128.unsqueeze(1).transpose(1, 3)


        save_image_2d(streak_prior_128, os.path.join(image_directory, "streak_prior_128.png"))

        diff_image = image - prior

        save_image_2d(diff_image, os.path.join(image_directory, "test_minus_prior.png"))
        corrected_image_128 = fbp_recon - streak_prior_128

        save_image_2d(corrected_image_128, os.path.join(image_directory, "corrected_image_128.png"))
        #test_vif = vifvec(image.squeeze().cpu().detach().numpy(), corrected_image_128.squeeze().cpu().detach().numpy())
        corrected_images = corrected_images.squeeze().cpu().detach().numpy()
        images = images.squeeze().cpu().detach().numpy()
        test_mse = mse(images, corrected_images)
        test_ssim = compare_ssim(images, corrected_images, axis=-1, data_range=1.0)
        test_psnr = psnr(images, corrected_images, data_range=1.0)

        test_vif = vif(image.squeeze().cpu().detach().numpy(), corrected_image_128.squeeze().cpu().detach().numpy())
        print(f"FINAL SSIM: {test_ssim}, MSE: {test_mse}, PSNR: {test_psnr}, VIF: {test_vif}")
