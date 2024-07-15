import os
import yaml
import numpy as np


import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data import ImageDataset_2D_Slices, ImageDataset_2D
from skimage.metrics import structural_similarity as compare_ssim


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def get_train_loader(dataset, batch_size, num_workers, sampler, pin_memory):   
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory)
    return loader

def get_data_loader_slices(img_path, img_dim, batch_size):    

    dataset = ImageDataset_2D_Slices(img_path, img_dim, batch_size)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return loader

def get_data_loader(img_path, img_dim, batch_size):    

    dataset = ImageDataset_2D(img_path, img_dim)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return loader

def get_data_loader_hdf5(dataset, batch_size):    

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return loader

def save_image_2d(tensor, file_name):
    '''
    tensor: [1, h, w, 1]
    '''
    tensor = tensor[0, ...].permute(2, 0, 1).cpu().data  # ([1, h, w, 1]) -> [1, h, w]
    image_grid = vutils.make_grid(tensor, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)

def reshape_tensor(previous_image, image):
    '''
    image: [1, h, w, 1]
    '''
    batch, height, width, channel = image.shape
    transform = T.Resize((height, width))                           # resize by interpolation (bilinear) ! not what i want , -> maybe padding instead?
    previous_image = transform(previous_image.permute(0, 3, 1, 2))  # ([1, x, y, 1]) -> [1, 1, x, y]  
    return previous_image.permute(0, 2, 3, 1)                       # ([1, 1, h, w]) -> [1, h, w, 1]

def get_image_pads(image_size, config):
    
    fbp_pad_xl = int((image_size[1][0] - 1))
    fbp_pad_xr = int((config['img_size'] - (image_size[0][0] - 1)))
    fbp_pad_yl = int((image_size[3][0] - 1))     
    fbp_pad_yr = int((config['img_size'] - (image_size[2][0] - 1)))    
     
    pads = [fbp_pad_xl, fbp_pad_xr, fbp_pad_yl, fbp_pad_yr]  
    return pads

def reshape_model_weights(image_height, image_width, config, checkpoint_directory):
        '''
        Load pretrain model weights and resize them to fit the new image shape
        '''
        # Load pretrain model
        model_path = os.path.join(checkpoint_directory, f"temp_model.pt")
        state_dict = torch.load(model_path)

        for weight in state_dict['net']: # reshape weights to fit new image\model size
            if 'weight' in weight: 
                if '.0.' in weight:                    
                    reshaped_weight = reshape_tensor(state_dict['net'][weight].unsqueeze(0).unsqueeze(3), torch.zeros(1, config['net']['network_width'], (image_height + image_width), 1))
                elif '.14.' in weight:
                    reshaped_weight = reshape_tensor(state_dict['net'][weight].unsqueeze(0).unsqueeze(3), torch.zeros(1, 1, config['net']['network_width'], 1))
                else:
                    reshaped_weight = reshape_tensor(state_dict['net'][weight].unsqueeze(0).unsqueeze(3), torch.zeros(1, config['net']['network_width'], config['net']['network_width'], 1))
            
                with torch.no_grad():
                    state_dict['net'][weight] = reshaped_weight.squeeze(3).squeeze(0) # reshape pretrain weights to fit new image size
                    
        return state_dict
                    
def shenanigans(skip, test_output, projectors, image, fbp_recon_128, train_projections, pads, it, iterations, image_directory, config): # image saving mumbo jumbo
    
    prior = test_output.cuda()
    image = image.cuda()

    if skip:        
        prior = reshape_tensor(test_output, image).cuda()
        projs_128 = projectors[1].forward_project(image.transpose(1, 3).squeeze(1))  # ([1, y, x])        -> [1, num_proj, x]
        fbp_recon_128 = projectors[1].backward_project(projs_128)                    # ([1, num_proj, x]) -> [1, y, x]
        fbp_recon_128 = fbp_recon_128.unsqueeze(1).transpose(1, 3)                   # ([1, y, x])        -> [1, x, y, 1]
        
   
    projs_prior_512 = projectors[0].forward_project(prior.transpose(1, 3).squeeze(1))  
    fbp_prior_512 = projectors[0].backward_project(projs_prior_512)  

    projs_prior_128 = projectors[1].forward_project(prior.transpose(1, 3).squeeze(1))  
    fbp_prior_128 = projectors[1].backward_project(projs_prior_128) 

    projs_prior_64 = projectors[2].forward_project(prior.transpose(1, 3).squeeze(1))
    fbp_prior_64 = projectors[2].backward_project(projs_prior_64)  
    
    streak_prior_128 = fbp_prior_128 - fbp_prior_512
    streak_prior_64 = fbp_prior_64 - fbp_prior_512

    fbp_prior_512 = fbp_prior_512.unsqueeze(1).transpose(1, 3)
    fbp_prior_128 = fbp_prior_128.unsqueeze(1).transpose(1, 3) 
    fbp_prior_64 = fbp_prior_64.unsqueeze(1).transpose(1, 3)  

    fbp_prior = torch.cat((fbp_prior_512, fbp_prior_128,  fbp_prior_64), 2)
    #save_image_2d(fbp_prior, os.path.join(image_directory, f"fbp_priors_{iterations + 1}_it_{it + 1}.png"))            

    streak_prior_64 = streak_prior_64.unsqueeze(1).transpose(1, 3) 
    streak_prior_128 = streak_prior_128.unsqueeze(1).transpose(1, 3) 
    streak_prior = torch.cat((streak_prior_128, streak_prior_64), 2)
    #save_image_2d(streak_prior, os.path.join(image_directory, f"streak_priors_{iterations + 1}_it_{it + 1}.png"))

    '''

    Compute Corrected image

    '''
    diff_image = image - prior
 
    corrected_image_128 = fbp_recon_128 - streak_prior_128 
    diff_corrected = image - corrected_image_128           

    diff_ssim_recon = compare_ssim(fbp_recon_128.transpose(1,3).squeeze().cpu().detach().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
    print(f"Diff SSIM recon = {diff_ssim_recon}")
    diff_ssim_train = compare_ssim(corrected_image_128.transpose(1,3).squeeze().cpu().detach().numpy(), image.transpose(1,3).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
    print(f"Diff SSIM train = {diff_ssim_train}")
    
    corrected_image_padded = F.pad(corrected_image_128, (0,0, pads[2],pads[3], pads[0],pads[1]))
    prior_padded = F.pad(prior, (0,0, pads[2],pads[3], pads[0],pads[1]))

    fbp_padded = F.pad(fbp_recon_128, (0,0, pads[2],pads[3], pads[0],pads[1])) 
    image_padded = F.pad(image, (0,0, pads[2],pads[3], pads[0],pads[1]))
    #print(f"train shape {train_projections.shape}")
    train_projections = train_projections.squeeze().unsqueeze(0)
                     # [1, num_proj, x, 1]
    train_pad = int((config['img_size'] - config['num_proj']) / 2)
    train_projections_padded = F.pad(train_projections, (0,0, train_pad,train_pad)).unsqueeze(3)    
    
    #print(f"im pa {image_padded.shape}, fb pa  ·{fbp_padded.shape} tr pa {train_projections_padded.shape}")
    input_image = torch.cat((image_padded, fbp_padded, (train_projections_padded / torch.max(train_projections_padded))), 2)    
    save_image_2d(input_image, os.path.join(image_directory, f"inputs_slice_{it +1}.png"))
    
    output_image =  torch.cat(((train_projections_padded / torch.max(train_projections_padded)), fbp_padded, prior_padded,  corrected_image_padded), 2)            
    save_image_2d(output_image, os.path.join(image_directory, f"outputs_slice_{it + 1}_iter_{iterations + 1}_SSIM_{diff_ssim_train}.png"))
    
    return train_projections, corrected_image_padded.cpu().detach().numpy()