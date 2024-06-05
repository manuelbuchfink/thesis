
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
model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}_v{}' \
.format(config['img_size'], config['num_proj_sparse_view_128'], config['model'], \
    config['net']['network_input_size'], config['net']['network_width'], \
    config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding'], datetime.now()))
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
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])


ct_projector_full_view_512 = FanBeam2DProjector(config['img_size'], config['proj_size'], config['num_proj_full_view_512'])
ct_projector_sparse_view_128 = FanBeam2DProjector(config['img_size'], config['proj_size'], config['num_proj_sparse_view_128'])
ct_projector_sparse_view_64 = FanBeam2DProjector(config['img_size'], config['proj_size'], config['num_proj_sparse_view_64'])

wandb.init(
    #set the wandb project where this run will be logged
    project="ct-image-reconstruction",

    # track hyperparameters and run metadata
    config={
    "learning_rate": config['lr'],
    "architecture": config['model'],
    "dataset": config['data'],
    "epochs": config['max_iter'],
    "fourier feature standard deviation" : config['encoder']['scale'],
    "img size" : config['img_size'],
    "batch size" : config['batch_size'],
    }
)
for it, (grid, image) in enumerate(data_loader):
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda()  # [bs, x, y, 3], [0, 1]
    image = image.cuda()  # [bs, x, y, 1], [0, 1]
    
    projs_128 = ct_projector_sparse_view_128.forward_project(image.transpose(1, 3).squeeze(1))  
    fbp_recon_128 = ct_projector_sparse_view_128.backward_project(projs_128) 
    
    for iterations in range(max_iter):
        train_loss_view = 0
        print(len(projs_128[0]))
        for view in range(len(projs_128[0])):        
            # print(projs_128[0][view])
            # print(projs_128[0][view].shape)
            model.train()
            optim.zero_grad()
            
            #train_embedding = encoder.embedding(view)
            train_output = model(projs_128[0][view])
            train_loss_view += 0.5 * loss_fn(train_output.to("cuda"), projs_128[0][view].to("cuda"))
            
        train_loss_view = 1/len(projs_128[0]) * train_loss_view
        train_loss_view.backward()
        optim.step()
        
    embedding = encoder.embedding(grid)    
    prior = model(embedding)
    
    print(prior.shape)
    save_image_2d(prior, os.path.join(image_directory, "prior.png"))