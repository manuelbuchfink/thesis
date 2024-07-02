
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
import torch.nn.functional as F
from data import ImageDataset_2D_hdf5
from networks import Positional_Encoder, FFN
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_image_2d
from skimage.metrics import structural_similarity as compare_ssim
import gc
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import time
import h5py
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

# Setup data loader
print('Load volume: {}'.format(config['img_path']))
dataset = ImageDataset_2D_hdf5(config['img_path'], config['img_size'], config['num_slices'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])


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



corrected_images = []
for it, (grid, image, image_size) in enumerate(data_loader):
    
    print(f"image size {image_size[0]}")
    image_size = int(image_size[0])
    
    ct_projector_full_view_512 = FanBeam2DProjector(image_size, image_size, config['num_proj_full_view_512'])
    ct_projector_sparse_view_128 = FanBeam2DProjector(image_size, image_size, config['num_proj_sparse_view_128'])
    ct_projector_sparse_view_64 = FanBeam2DProjector(image_size, image_size, config['num_proj_sparse_view_64'])
    
    # Setup input encoder:
    encoder = Positional_Encoder(config['encoder'], bb_embedding_size=image_size)

    # Setup model
    model = FFN(config['net'], image_size)
    model.cuda()
    model.train()
    
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    loss_fn = torch.nn.MSELoss().to("cuda")
    
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda()      # [1, x, y, 2], [0, 1]
    image = image.cuda()    # [1, x, y, 1], [0, 1]
    print(f"image shape {image.shape} grid shape {grid.shape}")
    
    projs_128 = ct_projector_sparse_view_128.forward_project(image.transpose(1, 3).squeeze(1))  # ([1, y, x])        -> [1, num_proj, x]
    fbp_recon_128 = ct_projector_sparse_view_128.backward_project(projs_128)                    # ([1, num_proj, x]) -> [1, y, x]

    projs_64 = ct_projector_sparse_view_64.forward_project(image.transpose(1, 3).squeeze(1))
    fbp_recon_64 = ct_projector_sparse_view_64.backward_project(projs_64)  
    
    train_proj128 = projs_128[..., np.newaxis]                  # [1, num_proj, x, 1]
    fbp_recon_128 = fbp_recon_128.unsqueeze(1).transpose(1, 3)  # [1, x, y, 1]
    fbp_recon_64 = fbp_recon_64.unsqueeze(1).transpose(1, 3) 
    
    train_pad = int((config['img_size'] - config['num_proj_sparse_view_128']) / 2)
    train_proj128 = F.pad(train_proj128, (0,0, 0,0, train_pad,train_pad))      
    
    
    fbp_pad = int((config['img_size'] - (image_size - 1)) / 2)
   
    fbp_padded = F.pad(fbp_recon_128, (0,0, fbp_pad,fbp_pad, fbp_pad,fbp_pad))
    print(f"fbp recon shape {fbp_recon_128.shape}, padding {fbp_pad}, fbp padded {fbp_padded.shape}")
    
    image_padded = F.pad(image, (0,0, fbp_pad,fbp_pad, fbp_pad,fbp_pad))
    input_image = torch.cat((image_padded, fbp_padded, train_proj128), 2)
    
    save_image_2d(input_image, os.path.join(image_directory, f"inputs_slice_{it}.png"))
    
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

        # Compute training psnr
        if (iterations + 1) % config['log_iter'] == 0:
            train_psnr = -10 * torch.log10(2 * train_loss).item()
            train_loss = train_loss.item()
            train_writer.add_scalar('train_loss', train_loss, iterations + 1)
            train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss, train_psnr))
        
        '''

        Compute Streak artifacts from prior image

        '''          
        if (iterations + 1) % config['val_iter'] == 0:

                    # get prior image once training is finished    
            model.eval()
            with torch.no_grad():
                test_output = train_output      # train model on grid
                test_loss = 0.5 * loss_fn(test_output.to("cuda"), image.to("cuda")) # compare grid with test image
                test_psnr = - 10 * torch.log10(2 * test_loss).item()
                test_loss = test_loss.item()
                #test_ssim = compare_ssim(test_output.transpose(1,3).squeeze().cpu().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)

            end = time.time()
            
            train_writer.add_scalar('test_loss', test_loss, iterations + 1)
            train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
            #save_image_2d(test_output, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(iterations + 1, test_psnr, test_ssim)))
            #print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g} | Time Elapsed {}".format(iterations + 1, max_iter, test_loss, test_psnr, test_ssim, (end - start)))
            #wandb.log({"ssim": test_ssim, "loss": test_loss, "psnr": test_psnr})
            
            prior = test_output
            
            projs_prior_512 = ct_projector_full_view_512.forward_project(prior.transpose(1, 3).squeeze(1))  
            fbp_prior_512 = ct_projector_full_view_512.backward_project(projs_prior_512)  

            projs_prior_128 = ct_projector_sparse_view_128.forward_project(prior.transpose(1, 3).squeeze(1))  
            fbp_prior_128 = ct_projector_sparse_view_128.backward_project(projs_prior_128) 

            projs_prior_64 = ct_projector_sparse_view_64.forward_project(prior.transpose(1, 3).squeeze(1))
            fbp_prior_64 = ct_projector_sparse_view_64.backward_project(projs_prior_64)  
            
            streak_prior_128 = fbp_prior_128 - fbp_prior_512
            streak_prior_64 = fbp_prior_64 - fbp_prior_512

            fbp_prior_512 = fbp_prior_512.unsqueeze(1).transpose(1, 3)
            fbp_prior_128 = fbp_prior_128.unsqueeze(1).transpose(1, 3) 
            fbp_prior_64 = fbp_prior_64.unsqueeze(1).transpose(1, 3)  

            fbp_prior = torch.cat((fbp_prior_512, fbp_prior_128,  fbp_prior_64), 2)
            #save_image_2d(fbp_prior, os.path.join(image_directory, f"fbp_priors_{iterations + 1}_it_{it}.png"))            

            streak_prior_64 = streak_prior_64.unsqueeze(1).transpose(1, 3) 
            streak_prior_128 = streak_prior_128.unsqueeze(1).transpose(1, 3) 
            streak_prior = torch.cat((streak_prior_128, streak_prior_64), 2)
            #save_image_2d(streak_prior, os.path.join(image_directory, f"streak_priors_{iterations + 1}_it_{it}.png"))
        
            '''

            Compute Corrected image

            '''
            diff_image = image - prior
            corrected_image_128 = fbp_recon_128 - streak_prior_128 
            diff_corrected = image - corrected_image_128           
            #diff_ssim = compare_ssim(corrected_image_128.transpose(1,3).squeeze().cpu().detach().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
            #print(f"Diff SSIM = {diff_ssim}")
            
            corrected_images.append(corrected_image_128.cpu().detach().numpy())
            output_image =  torch.cat((prior, fbp_recon_128, corrected_image_128), 2)
            save_image_2d(output_image, os.path.join(image_directory, f"outputs_{iterations + 1}_slice_{it}.png"))
            
    
    # new_image_path = os.path.join(image_directory, '_corrected.h5')   
    # with h5py.File(new_image_path,'w') as h5f:
    #     h5f.create_dataset("VolumeCorrected", data=np.asarray(corrected_images))
        
    # Save final model            
    model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
    torch.save({'net': model.state_dict(), \
                'enc': encoder.B, \
                'opt': optim.state_dict(), \
                }, model_name)
        

