
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
import torch
import torch.backends.cudnn as cudnn
import tensorboardX
from ct_2d_projector import FanBeam2DProjector
import numpy as np

from networks import Positional_Encoder, FFN
from utils import get_config, prepare_sub_folder, get_data_loader, save_image_2d
from skimage.metrics import structural_similarity as compare_ssim
import gc
from datetime import datetime
import warnings
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


output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}_v{}_{}' \
.format(config['img_size'], config['num_proj_sparse_view_128'], config['model'], \
    config['net']['network_input_size'], config['net']['network_width'], \
    config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding'], datetime.now(), config['description']))
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


# Setup input encoder:
encoder = Positional_Encoder(config['encoder'])

# Setup model
model = FFN(config['net'])
model.cuda()
model.train()

optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
loss_fn = torch.nn.MSELoss().to("cuda")

# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['img_path'], config['img_size'], batch_size=config['batch_size'])

ct_projector_full_view_512 = FanBeam2DProjector(config['img_size'], config['proj_size'], config['num_proj_full_view_512'])
ct_projector_sparse_view_128 = FanBeam2DProjector(config['img_size'], config['proj_size'], config['num_proj_sparse_view_128'])

for it, (grid, image) in enumerate(data_loader):
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda()      # [1, x, y, 2], [0, 1]
    image = image.cuda()    # [1, x, y, 1], [0, 1]

    projs_128 = ct_projector_sparse_view_128.forward_project(image.transpose(1, 3).squeeze(1))  # ([1, y, x])        -> [1, num_proj, x]
    fbp_recon_128 = ct_projector_sparse_view_128.backward_project(projs_128)                    # ([1, num_proj, x]) -> [1, y, x]
    
    train_proj128 = projs_128[..., np.newaxis]                  # [1, num_proj, x, 1]
    fbp_recon_128 = fbp_recon_128.unsqueeze(1).transpose(1, 3)  # [1, x, y, 1]    

    save_image_2d(image, os.path.join(image_directory, "test.png"))
    save_image_2d(train_proj128, os.path.join(image_directory, "train128.png"))
    save_image_2d(fbp_recon_128, os.path.join(image_directory, "fbprecon_128.png"))
    
    train_embedding = encoder.embedding(grid)  #  fourier feature embedding:  ([1, x, y, 2] * [2, embedding_size]) -> [1, x, y, embedding_size]   
    
    # Train model
    for iterations in range(max_iter):

        model.train()
        optim.zero_grad()
    
        train_output = model(train_embedding)      #  train model on grid:        ([1, x, y, embedding_size])          -> [1, x, y, 1]

        train_projs = ct_projector_sparse_view_128.forward_project(train_output.transpose(1, 3).squeeze(1)).to("cuda")      # evaluate by forward projecting
        train_loss = (0.5 * loss_fn(train_projs.to("cuda"), projs_128.to("cuda")))                                          # compare forward projected grid with sparse view projection
     
        train_loss.backward()
        optim.step()     
        
                  
    '''

    Compute Prior images

    '''         
    prior = train_output 
    save_image_2d(prior, os.path.join(image_directory, f"prior_{iterations + 1}.png"))            
    
    projs_prior_512 = ct_projector_full_view_512.forward_project(prior.transpose(1, 3).squeeze(1))  
    fbp_prior_512 = ct_projector_full_view_512.backward_project(projs_prior_512)  

    projs_prior_128 = ct_projector_sparse_view_128.forward_project(prior.transpose(1, 3).squeeze(1))  
    fbp_prior_128 = ct_projector_sparse_view_128.backward_project(projs_prior_128) 

    
    streak_prior_128 = fbp_prior_128 - fbp_prior_512

    fbp_prior_512 = fbp_prior_512.unsqueeze(1).transpose(1, 3)
    fbp_prior_128 = fbp_prior_128.unsqueeze(1).transpose(1, 3)  

    save_image_2d(fbp_prior_512, os.path.join(image_directory, f"fbp_prior_512_{iterations + 1}.png"))
    save_image_2d(fbp_prior_128, os.path.join(image_directory, f"fbp_prior_128_{iterations + 1}.png"))


    streak_prior_128 = streak_prior_128.unsqueeze(1).transpose(1, 3) 

    save_image_2d(streak_prior_128, os.path.join(image_directory, f"streak_prior_128_{iterations + 1}.png"))


    '''

    Compute Corrected image

    '''
    diff_image = image - prior;
    save_image_2d(diff_image, os.path.join(image_directory, f"test_minus_prior_{iterations + 1}.png"))
    corrected_image_128 = fbp_recon_128 - streak_prior_128

    save_image_2d(corrected_image_128, os.path.join(image_directory, f"corrected_image_128_{iterations + 1}.png"))

    diff_corrected = image - corrected_image_128
    save_image_2d(diff_corrected, os.path.join(image_directory, f"diff_corrected_image_128_{iterations + 1}.png"))
    print(f"correted image shape {corrected_image_128.shape}, image shape {image.shape}")
    diff_ssim = compare_ssim(corrected_image_128.transpose(1,3).squeeze().cpu().detach().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
    print(f"Diff SSIM = {diff_ssim}")
    
    # Save final model            
    model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
    torch.save({'net': model.state_dict(), \
                'enc': encoder.B, \
                'opt': optim.state_dict(), \
                }, model_name)
        

