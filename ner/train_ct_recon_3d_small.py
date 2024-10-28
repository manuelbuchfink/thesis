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

from networks import Positional_Encoder_3D, FFN_3D
from ct_3d_projector import ConeBeam3DProjector
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_volume, save_image_2d
from data import ImageDataset_3D_hdf5

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error

from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from skimage.metrics import mean_squared_error  as mse # pylint: disable=import-error
from skimage.metrics import peak_signal_noise_ratio as psnr

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

# Setup data loader
print('Load volume: {}'.format(config['img_path']))
dataset = ImageDataset_3D_hdf5(config['img_path'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

for it, (grid, image) in enumerate(data_loader):

    n = config['down_sample_factor']
    image = image.squeeze()[::n,::n,::n].unsqueeze(0).unsqueeze(4).cuda()    # [1, h, w, d, 1], value range = [0, 1]
    grid = grid.squeeze()[::n,::n,::n].unsqueeze(0).cuda()                   # [1, h, w, d, 3], value range = [0, 1]

    output_subfolder = config['data']
    model_name = os.path.join(output_folder, f"{output_subfolder}/img{list(image.squeeze().shape)}_proj{config['num_proj_sparse_view']}_{config['lr']}_scale{config['encoder']['scale']}_size{config['encoder']['embedding_size']}")
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    print(model_name)

    ct_projector_sparse_view = ConeBeam3DProjector(image.squeeze().shape, num_proj=config['num_proj_sparse_view'])

    projections = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
    fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

    fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4)                                          # [1, h, w, 1]
    fbp_recon = torch.tensor(fbp_recon, dtype=torch.float16)                                    # [B, C, H, W]
    fbp_volume = torch.tensor(fbp_recon, dtype=torch.float16)


    # Setup input encoder:
    encoder = Positional_Encoder_3D(config['encoder'])

    # Setup model
    model = FFN_3D(config['net'])

    model.cuda()
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    loss_fn = torch.nn.MSELoss().cuda()

    train_embedding = encoder.embedding(grid)  # fourier feature embedding:  ([1, x, y, z, 3] * [3, embedding_size]) -> [1, z, x, y, embedding_size]
    scaler = torch.cuda.amp.GradScaler()

    for iterations in range(max_iter): # train model

        model.train()
        optim.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            train_output = model(train_embedding)                                           # train model on grid: ([1, x, y, embedding_size]) > [1, x, y, 1]
            train_loss = (0.5 * loss_fn(train_output.to("cuda"), fbp_recon.to("cuda")))     # compare forward projected grid with sparse view projection

        train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 4).squeeze(1)).to("cuda")      # evaluate by forward projecting

        scaler.scale(train_loss).backward()
        scaler.step(optim)
        scaler.update()

        if (iterations + 1) % config['val_iter'] == 0: # compute metrics
            save_image_2d(train_output[:,87,:,:,:].float(), os.path.join(image_directory, f"test_slice_{iterations + 1}.png"))
            model.eval()
            with torch.no_grad():
                fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 4)
                test_ssim = compare_ssim(fbp_prior.transpose(1,4).squeeze().cpu().numpy(), fbp_recon.transpose(1,4).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
                test_mse = mse(train_projections.squeeze().cpu().numpy(), projections.squeeze().cpu().numpy())
                test_psnr = psnr(train_projections.squeeze().cpu().numpy(), projections.squeeze().cpu().numpy(), data_range=1.0)

            end = time.time()

            print("[Volume Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | MSE {:.4g} | PSNR {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, test_mse, test_psnr, (end - start) / 60))

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    LOAD IMAGE SLICES INTO CORRECTED_IMAGES


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    prior_volume = train_output.squeeze()
    prior_volume = torch.tensor(prior_volume, dtype=torch.float16)[None, ...].unsqueeze(0)
    prior_volume = prior_volume.squeeze(0).unsqueeze(4)

    fbp_volume = fbp_volume.squeeze()
    fbp_volume = torch.tensor(fbp_volume, dtype=torch.float16)[None, ...]

    ct_projector_full_view = ConeBeam3DProjector(fbp_volume.squeeze().shape, 512)
    ct_projector_sparse_view = ConeBeam3DProjector(fbp_volume.squeeze().shape, num_proj=config['num_proj_sparse_view'])

    projs_prior_full_view = ct_projector_full_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
    fbp_prior_full_view = ct_projector_full_view.backward_project(projs_prior_full_view)
    fbp_prior_full_view = fbp_prior_full_view.unsqueeze(1).transpose(1, 4)

    projs_prior_sparse_view = ct_projector_sparse_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
    fbp_prior_sparse_view = ct_projector_sparse_view.backward_project(projs_prior_sparse_view)
    fbp_prior_sparse_view = fbp_prior_sparse_view.unsqueeze(1).transpose(1, 4)

    streak_volume = (fbp_prior_sparse_view - fbp_prior_full_view)
    corrected_volume = (fbp_volume.unsqueeze(4) - streak_volume).squeeze().cpu().detach().numpy()

    fbp_prior_full_view = fbp_prior_full_view.squeeze().cpu().detach().numpy()
    fbp_volume = fbp_volume.squeeze().cpu().detach().numpy()
    prior_volume = prior_volume.squeeze().cuda().cpu().detach().numpy()
    streak_volume = streak_volume.squeeze().cuda().cpu().detach().numpy()
    image = image.squeeze().cpu().detach().numpy()

    test_mse = mse(image, corrected_volume)
    test_ssim = compare_ssim(image, corrected_volume, axis=-1, data_range=1.0)
    test_psnr = psnr(image, corrected_volume, data_range=1.0)

    print(f"FINAL SSIM: {test_ssim}, MSE: {test_mse}, PSNR: {test_psnr}")

    save_volume(fbp_volume, image_directory, config, "fbp_volume")
    save_volume(corrected_volume, image_directory, config, "corrected_volume")
    save_volume(prior_volume, image_directory, config, "prior_volume")
    save_volume(streak_volume, image_directory, config, "streak_volume")
    save_volume(fbp_prior_full_view, image_directory, config, "fbp_prior_full_view")