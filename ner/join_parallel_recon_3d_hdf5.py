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

from utils import get_config, get_sub_folder, save_volume
from ct_3d_projector import ConeBeam3DProjector

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error
import torch.nn.functional as F # pylint: disable=import-error

from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from skimage.metrics import mean_squared_error  as mse # pylint: disable=import-error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import profile_line
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)

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


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

LOAD IMAGE SLICES INTO CORRECTED_IMAGES AND SPARSE_IMAGES FROM FILE


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
priors = []
for i in range(4):

    prior_volume = torch.load(os.path.join(image_directory, f'prior_volume_{i}.pt'))
    prior_volume = prior_volume.squeeze()
    prior_volume = torch.tensor(prior_volume, dtype=torch.float16)[None, ...]
    priors.append(prior_volume.squeeze(0))

prior_volume  = torch.cat(priors, 0).unsqueeze(0)

image = torch.load(os.path.join(image_directory, f'image.pt')).squeeze().cpu().float().detach().numpy()

fbp_volume = torch.load(os.path.join(image_directory, f'fbp_volume.pt'))
fbp_volume = fbp_volume.squeeze()
fbp_volume = torch.tensor(fbp_volume, dtype=torch.float16)[None, ...]

ct_projector_full_view = ConeBeam3DProjector(fbp_volume.squeeze().shape, num_proj=512)
ct_projector_sparse_view = ConeBeam3DProjector(fbp_volume.squeeze().shape, num_proj=config['num_proj_sparse_view'])

prior_volume = prior_volume.unsqueeze(4)

projs_prior_full_view = ct_projector_full_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
fbp_prior_full_view = ct_projector_full_view.backward_project(projs_prior_full_view)

projs_prior_sparse_view = ct_projector_sparse_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
fbp_prior_sparse_view = ct_projector_sparse_view.backward_project(projs_prior_sparse_view)

streak_volume = (fbp_prior_sparse_view - fbp_prior_full_view).unsqueeze(1).transpose(1, 4)

corrected_volume = (fbp_volume.unsqueeze(4) - streak_volume).squeeze().cpu().detach().numpy()

fbp_prior_full_view = fbp_prior_full_view.squeeze().cpu().float().detach().numpy()
fbp_volume = fbp_volume.squeeze().cpu().float().detach().numpy()
prior_volume = prior_volume.squeeze().cuda().cpu().float().detach().numpy()
streak_volume = streak_volume.squeeze().cuda().cpu().float().detach().numpy()

#save_volume(fbp_prior_full_view, image_directory, config, "fbp_prior_full_view")
save_volume(fbp_volume, image_directory, config, "fbp_volume")
save_volume(corrected_volume, image_directory, config, "corrected_volume")
save_volume(prior_volume, image_directory, config, "prior_volume")
save_volume(streak_volume, image_directory, config, "streak_volume")

test_mse = mse(image, corrected_volume)
test_ssim = compare_ssim(image, corrected_volume, axis=-1, data_range=1.0)
test_psnr = psnr(image, corrected_volume, data_range=1.0)
print(f"FINAL SSIM: {test_ssim}, MSE: {test_mse}, PSNR: {test_psnr}")