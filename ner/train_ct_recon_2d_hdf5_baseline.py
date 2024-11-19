
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
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ct_2d_projector_baseline import FanBeam2DProjector
from networks import Positional_Encoder_base, FFN
from data import ImageDataset_2D_sparsify
from utils import get_config, get_data_loader_hdf5, save_volume, compute_vif, prepare_sub_folder, save_image_2d

from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from skimage.metrics import mean_squared_error  as mse # pylint: disable=import-error
from skimage.metrics import peak_signal_noise_ratio as psnr


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
dataset = ImageDataset_2D_sparsify(config['img_path'], parser)
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

corrected_images = []
prior_images = []
fbp_images = []
images = []
for it, (grid, image) in enumerate(data_loader):
    if it > 254 and it < 256:
        # Input coordinates (h,w) grid and target image
        grid = grid.cuda()      # [1, h, w, 2], value range = [0, 1]
        image = image.cuda()    # [1, h, w, 1], value range = [0, 1]

        ct_projector_full_view = FanBeam2DProjector(image.squeeze().shape[1], image_width=image.squeeze().shape[0], proj_size=config['proj_size'], num_proj=config['num_proj_full_view'])
        ct_projector_sparse_view = FanBeam2DProjector(image_height=image.squeeze().shape[1], image_width=image.squeeze().shape[0], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])
        projectors = [ct_projector_full_view, ct_projector_sparse_view]

        projections = ct_projector_sparse_view.forward_project(image.transpose(1, 3).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
        fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

        train_projections = projections[..., np.newaxis]                                             # [1, num_proj_sparse_view, original_image_size, 1]
        fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 3)                                          # [1, h, w, 1]

        # Setup input encoder:
        encoder = Positional_Encoder_base(config['encoder'], bb_embedding_size= int(image.squeeze().shape[1] + image.squeeze().shape[0]))

        # Setup model
        model = FFN(config['net'], int(image.squeeze().shape[1] + image.squeeze().shape[0]))

        model.cuda()
        model.train()

        optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
        loss_fn = torch.nn.MSELoss().cuda()

        train_embedding = encoder.embedding(grid)   # fourier feature embedding:  ([1, x, y, 2] * [2, embedding_size]) -> [1, x, y, embedding_size]

        # Train model
        for iterations in range(max_iter):

            model.train()
            optim.zero_grad()

            train_output = model(train_embedding)  # train model on grid: ([1, x, y, embedding_size]) > [1, x, y, 1]

            train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 3).squeeze(1)).to("cuda")      # evaluate by forward projecting
            train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))                                    # compare forward projected grid with sparse view projection
            #train_loss = (0.5 * loss_fn(train_output.to("cuda"), fbp_recon.to("cuda")))     # compare forward projected grid with sparse view projection

            # Compute ssim
            if (iterations + 1) % config['val_iter'] == 0:
                if config['eval']:
                    model.eval()
                    with torch.no_grad():
                        test_loss = 0.5 * loss_fn(train_output.to("cuda"), image.to("cuda")) # compare grid with test image
                        test_psnr = - 10 * torch.log10(2 * test_loss).item()
                        test_loss = test_loss.item()

                        test_ssim = compare_ssim(train_output.transpose(1,3).squeeze().cpu().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
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
                        test_ssim = compare_ssim(fbp_prior.transpose(1,3).squeeze().cpu().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
                        fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 3)
                        test_ssim = compare_ssim(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), multichannel=True, data_range=1.0)
                        test_mse = mse(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy())
                        test_psnr = psnr(fbp_prior.transpose(1,3).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), data_range=1.0)

                    end = time.time()
                    print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | MSE {:.4g} | PSNR {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, test_mse, test_psnr, (end - start) / 60))

            train_loss.backward()
            optim.step()

        prior = train_output

        projs_prior_full_view = projectors[0].forward_project(prior.transpose(1, 3).squeeze(1))
        fbp_prior_full_view = projectors[0].backward_project(projs_prior_full_view)

        projs_prior_sparse_view = projectors[1].forward_project(prior.transpose(1, 3).squeeze(1))
        fbp_prior_sparse_view = projectors[1].backward_project(projs_prior_sparse_view)

        streak_prior = (fbp_prior_sparse_view - fbp_prior_full_view).unsqueeze(1).transpose(1, 3)
        fbp_prior_sparse_view = fbp_prior_sparse_view.unsqueeze(1).transpose(1, 3)

        corrected_image = fbp_recon - streak_prior

        diff_ssim_recon = compare_ssim(fbp_recon.transpose(1,3).squeeze().cpu().detach().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
        diff_ssim_train = compare_ssim(corrected_image.transpose(1,3).squeeze().cpu().detach().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)

        print(f"Diff SSIM TRAIN = {diff_ssim_train}, Diff SSIM RECON = {diff_ssim_recon}")

        print(f"sadasd {corrected_image.shape} {prior.shape} {fbp_recon.shape}")
        save_image_2d(fbp_recon, os.path.join(checkpoint_directory, f"FBP_volume.png"))
        save_image_2d(corrected_image, os.path.join(checkpoint_directory, f"corrected_volume.png"))
        save_image_2d(prior, os.path.join(checkpoint_directory, f"prior_volume.png"))

        corrected_images.append(corrected_image)
        prior_images.append(prior)
        fbp_images.append(fbp_recon)
        images.append(image)



images = torch.cat(images, 0)
corrected_images = torch.cat(corrected_images, 0)
prior_images = torch.cat(prior_images, 0).squeeze().cpu().detach().numpy()
fbp_images = torch.cat(fbp_images, 0).squeeze().cpu().detach().numpy()



#save_image_2d(streak_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"streak_volume_prior_difference.png"))

#test_vif = compute_vif(images, corrected_images)

corrected_images = corrected_images.squeeze().cpu().detach().numpy()
images = images.squeeze().cpu().detach().numpy()
test_mse = mse(images, corrected_images)
test_ssim = compare_ssim(images, corrected_images, axis=-1, data_range=1.0)
test_psnr = psnr(images, corrected_images, data_range=1.0)


print(f"FINAL SSIM: {test_ssim}, MSE: {test_mse}, PSNR: {test_psnr}")#, VIF: {test_vif}")

# save_volume(fbp_images, image_directory, config, "fbp_volume")
# save_volume(corrected_images, image_directory, config, "corrected_volume")
# save_volume(prior_images, image_directory, config, "prior_volume")
with open('resultmetrics', 'a+')as file:
    file.write(f"{test_ssim}, {test_psnr}, {(end - start) / 60}, ")