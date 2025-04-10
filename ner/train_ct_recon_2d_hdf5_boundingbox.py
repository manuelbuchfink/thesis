
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
import torch.nn.functional as F

from ct_2d_projector_baseline import FanBeam2DProjector
from networks import Positional_Encoder_base, FFN
from data import ImageDataset_2D_hdf5_canny_baseline
from utils import get_config, get_data_loader_hdf5, save_volume, save_image_2d, compute_vif, prepare_sub_folder, get_image_pads

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
dataset = ImageDataset_2D_hdf5_canny_baseline(config['img_path'], parser, image_directory)
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

corrected_images = []
prior_images = []
fbp_images = []
images = []
for it, (grid, image, image_size) in enumerate(data_loader):
    if it > 254 and it < 256:
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
        print(f"image height and width {image_height} {image_width}")

        if image_height == 0 or image_width == 0: # skip emty images

            skip_image = torch.zeros(1, 512, 512).cuda()
            corrected_images.append(skip_image)
            prior_images.append(skip_image)
            fbp_images.append(skip_image)
            images.append(skip_image)

            continue

        elif image_height < 7: # pad small image slices so that SSIM can be computed
            height_pad = (7 - image_height)
            grid = F.pad(grid, (0,0, 0,0, height_pad,height_pad))
            image = F.pad(image, (0,0, 0,0, height_pad,height_pad))
            image_height += ((height_pad) * 2)

        elif image_width < 7: # pad small image slices so that SSIM can be computed
            width_pad = (7 - image_width)
            grid = F.pad(grid, (0,0, width_pad,width_pad, 0,0))
            image = F.pad(image, (0,0, width_pad,width_pad, 0,0))
            image_width += ((width_pad) * 2)

        ct_projector_full_view = FanBeam2DProjector(image_height=image_height, image_width=image_width, proj_size=config['proj_size'], num_proj=config['num_proj_full_view'])
        ct_projector_sparse_view = FanBeam2DProjector(image_height=image_height, image_width=image_width, proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])
        projectors = [ct_projector_full_view, ct_projector_sparse_view]

        projections = ct_projector_sparse_view.forward_project(image.transpose(1, 3).squeeze(1))     # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
        fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

        train_projections = projections[..., np.newaxis]                                           # [1, num_proj_sparse_view, original_image_size, 1]
        fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 3)                                           # [1, h, w, 1]

        # Setup input encoder:
        encoder = Positional_Encoder_base(config['encoder'], bb_embedding_size= int(image_height + image_width))

        # Setup model
        model = FFN(config['net'], int(image_height + image_width))

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

        corrected_image = F.pad(corrected_image.squeeze(), (pads[2],pads[3], pads[0],pads[1])).unsqueeze(0)
        prior = F.pad(prior.squeeze(), (pads[2],pads[3], pads[0],pads[1])).unsqueeze(0)
        fbp_recon = F.pad(fbp_recon.squeeze(), (pads[2],pads[3], pads[0],pads[1])).unsqueeze(0)
        image = F.pad(image.squeeze(), (pads[2],pads[3], pads[0],pads[1])).unsqueeze(0)

        save_image_2d(corrected_image.squeeze().float().unsqueeze(0).unsqueeze(3), os.path.join(image_directory, f"corrected_image_{it}.png"))
        save_image_2d(fbp_recon.squeeze().float().unsqueeze(0).unsqueeze(3), os.path.join(image_directory, f"fbp_image_{it}.png"))
        save_image_2d(prior.squeeze().float().unsqueeze(0).unsqueeze(3), os.path.join(image_directory, f"prior_image_{it}.png"))
        corrected_images.append(corrected_image)
        prior_images.append(prior)
        fbp_images.append(fbp_recon)
        images.append(image)

images = torch.cat(images, 0)
corrected_images = torch.cat(corrected_images, 0)
prior_images = torch.cat(prior_images, 0).squeeze().cpu().detach().numpy()
fbp_images = torch.cat(fbp_images, 0).squeeze().cpu().detach().numpy()


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