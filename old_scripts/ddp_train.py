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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from data import display_tensor_stats
import tensorboardX
from ct_2d_projector import FanBeam2DProjector
from ct_2d_iterative_projector import FanBeam2DProjectorIterative
import numpy as np
from data import ImageDataset_2D_Slices
from networks import Positional_Encoder, FFN
from utils import get_config, prepare_sub_folder, get_train_loader, save_image_2d
from skimage.metrics import structural_similarity as compare_ssim

import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
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


ct_projector_full_view_512 = FanBeam2DProjectorIterative(config['img_size'], config['proj_size'], config['num_proj_full_view_512'], config['batch_size'])
ct_projector_sparse_view_128_iter = FanBeam2DProjectorIterative(config['img_size'], config['proj_size'], config['num_proj_sparse_view_128'], config['batch_size'])
ct_projector_sparse_view_64_iter = FanBeam2DProjectorIterative(config['img_size'], config['proj_size'], config['num_proj_sparse_view_64'], config['batch_size'])

embeddings = [None] * config['batch_size']
projections_64 = [None] * config['batch_size']
projections_128 = [None] * config['batch_size']
images = [None] * config['batch_size']
grids = [None] * config['batch_size']
def init_projections(device, train_loader):
    
    for it, (grid, image) in enumerate(train_loader):
        
        # Input coordinates (x,y) grid and target image
        print(len(grids))
        grid = grid.to(device)     # [bs, x/bs, y, 2], [0, 1]
        grids[device] = grid
        
        image = image.to(device)    # [bs, x/bs, y, 1], [0, 1]
        images[device] = image
        
        projections_128[device] = ct_projector_sparse_view_128_iter.forward_project(image.transpose(1, 3).squeeze(1))  # ([1, y, x])        -> [1, num_proj, x]
        fbp_recon_128 = ct_projector_sparse_view_128_iter.backward_project(projections_128[device])                           # ([1, num_proj, x]) -> [1, y, x]

        # projections_64.append(ct_projector_sparse_view_64_iter.forward_project(images[it].transpose(1, 3).squeeze(1)))
        # fbp_recon_64 = ct_projector_sparse_view_64_iter.backward_project([projections_64[it]])  
        
        train_proj128 = projections_128[device][..., np.newaxis]        # [1, num_proj, x, 1]
        fbp_recon_128 = fbp_recon_128.unsqueeze(1).transpose(1, 3)  # [1, x, y, 1]
        #fbp_recon_64 = fbp_recon_64.unsqueeze(1).transpose(1, 3)     

        save_image_2d(image, os.path.join(image_directory, f"test_{device}.png"))
        save_image_2d(train_proj128, os.path.join(image_directory, f"train128_{device}.png"))
        save_image_2d(fbp_recon_128, os.path.join(image_directory, f"fbprecon_128{device}.png"))

        embeddings[device] = encoder.embedding(grid)  #  fourier feature embedding:  ([1, x, y, 2] * [2, embedding_size]) -> [1, x, y, embedding_size]
        
        print(f"grid: {grid.shape}")

    
def train(model, device, train_loader, optim, iterations):
    model.train()
    for it, (grid, image) in enumerate(train_loader):       
        print("hello")
        
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main():
    # Training settings
    
    use_cuda = torch.cuda.is_available()    
    train_kwargs = {'batch_size': config['batch_size']}   
    if use_cuda:
        cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)        

 
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")
    print(f"batch size: {config['batch_size']}")
    dataset = ImageDataset_2D_Slices(img_path=config['img_path'], img_dim=config['img_size'], batch_size=config['batch_size']) #image slices
    # Display image and label.

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    

    data_loader = get_train_loader(dataset=dataset,
                                  sampler=train_sampler,
                                  pin_memory=True,  
                                  batch_size=config['batch_size'],
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),                                                                    
                                  )
    optims = []
    for it, (grid, image) in enumerate(data_loader): 
        print(f"it: {it}, grid: {grid.shape}, image: {display_tensor_stats(image)}")
        optims.append(optim.Adam(ddp_model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay']))
        
    model = FFN(config['net'])
    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(ddp_model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    init_projections(train_loader=data_loader, device=local_rank)

    print(f"optim length: {len(optims)}")
    for iterations in range(max_iter):
        train(ddp_model, local_rank, data_loader, optimizer, iterations)      
        scheduler.step()
        
    if rank == 0:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        
    dist.destroy_process_group()
    

if __name__ == '__main__':
    main()