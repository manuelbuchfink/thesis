import os
import yaml
import numpy as np
import h5py
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
# img_ref, img_dist are two images of the same size.
from vif_utils import vif
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

def get_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')

    return checkpoint_directory, image_directory

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

def save_image(tensor, file_name):
    '''
    tensor: [1, h, w]
    '''
    image_grid = vutils.make_grid(tensor, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)

def get_image_pads(image_size, config):
    '''
    rt: top of center  rb: bottom of center
    cl: left of center   cr: right of center
    '''
    fbp_pad_rt = int((image_size[1][0] - 1))
    fbp_pad_rb = int((config['img_size'] - (image_size[0][0] - 1)))
    fbp_pad_cl = int((image_size[3][0] - 1))
    fbp_pad_cr = int((config['img_size'] - (image_size[2][0] - 1)))

    return [fbp_pad_rt, fbp_pad_rb, fbp_pad_cl, fbp_pad_cr]

def compute_vif(ground_truth, corrected_image):
    '''
    Visual Information Fidelity of two volumes
    '''
    avg_vif = 0

    slices = int(ground_truth.squeeze().shape[0])
    print(f"slices {slices} ground meat {ground_truth.shape} the other {corrected_image.shape}")

    ground_truth = ground_truth.squeeze().cpu().detach().numpy()
    corrected_image = corrected_image.squeeze().cpu().detach().numpy()
    for i in range(0, slices, 1):
        avg_vif += vif(ground_truth[i], corrected_image[i])

    return avg_vif / slices

def reshape_tensor(previous_image, image):
    '''
    image: [1, h, w, 1]
    '''
    batch, height, width, channel = image.shape
    transform = T.Resize((height, width))                           # resize by interpolation (bilinear)
    previous_image = transform(previous_image.permute(0, 3, 1, 2))  # ([1, h, w, 1]) -> [1, 1, h, w]
    return previous_image.permute(0, 2, 3, 1)                       # ([1, 1, h, w]) -> [1, h, w, 1]

def reshape_model_weights(image_height, image_width, config, checkpoint_directory, id):
        '''
        Load pretrain model weights and resize them to fit the new image shape
        '''
        # Load pretrain model
        model_path = os.path.join(checkpoint_directory, f"temp_model_{id}.pt")
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

def save_volume(volume, image_directory, config, name):


    # save corrected slices in new hdf5 Volume
    volume_path = os.path.join(image_directory, f"../{config['data'][:-5]}_{name}_with_{config['num_proj_sparse_view']}_projections.hdf5")
    print(f"saved to {config['data'][:-5]}_{name}_with_{config['num_proj_sparse_view']}_projections.hdf5")

    gridSpacing=[5.742e-05, 5.742e-05, 5.742e-05]
    gridOrigin=[0, 0 ,0]
    if not(os.path.isdir(f"./u_{name}/")):
        os.mkdir(f"./u_{name}/")

    with h5py.File(volume_path,'w') as hdf5:
        hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
        hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
        hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
        hdf5.create_dataset("Volume", data=np.asarray(volume))


    volume = h5py.File(volume_path, 'r')
    volume = volume['Volume']
    slices_sparse = [None] * int(volume.shape[0])
    for i in range(int(volume.shape[0])):

        #split image into N evenly sized chunks
        slices_sparse[i] = volume[i,:,:].squeeze()           # (512,512) = [h, w]
        save_image(torch.tensor(slices_sparse[i], dtype=torch.float32), f"./u_{name}/image from saved volume, slice Nr. {i}.png")
    print(f"image: {name} saved")

def correct_image_slice_all(skip, test_output, projectors, image, fbp_recon, train_projections, pads, it, image_directory, config): # image saving mumbo jumbo
    '''

    Compute Corrected image

    '''
    image = image.cuda()                                                                 # [1, h, w, 1]
    prior = (reshape_tensor(test_output, image).cuda() if skip else test_output.cuda())  # [1, h, w, 1]

    projs_prior_full_view = projectors[0].forward_project(prior.transpose(1, 3).squeeze(1))
    fbp_prior_full_view = projectors[0].backward_project(projs_prior_full_view)

    projs_prior_sparse_view = projectors[1].forward_project(prior.transpose(1, 3).squeeze(1))
    fbp_prior_sparse_view = projectors[1].backward_project(projs_prior_sparse_view)

    streak_prior = (fbp_prior_sparse_view - fbp_prior_full_view).unsqueeze(1).transpose(1, 3)
    fbp_prior_sparse_view = fbp_prior_sparse_view.unsqueeze(1).transpose(1, 3)

    corrected_image = fbp_recon - streak_prior


    corrected_image_padded = F.pad(corrected_image, (0,0, pads[2],pads[3], pads[0],pads[1]))

    prior_image_padded = F.pad(prior, (0,0, pads[2],pads[3], pads[0],pads[1]))
    torch.save(corrected_image_padded, os.path.join(image_directory, f"corrected_slice_{it + 1}.pt"))
    torch.save(prior_image_padded, os.path.join(image_directory, f"prior_slice_{it + 1}.pt"))

    #fbp_padded = F.pad(fbp_recon, (0,0, pads[2],pads[3], pads[0],pads[1]))
    # prior_padded = F.pad(prior, (0,0, pads[2],pads[3], pads[0],pads[1]))
    # image_padded = F.pad(image, (0,0, pads[2],pads[3], pads[0],pads[1]))

    train_projections = train_projections.squeeze().unsqueeze(0)
    train_pad = int((config['img_size'] - config['num_proj_sparse_view']) / 2)
    train_projections_padded = F.pad(train_projections, (0,0, train_pad,train_pad)).unsqueeze(3)

    return corrected_image, prior, train_projections

def correct_image_slice(test_output, projectors, fbp_recon, train_projections): # image saving mumbo jumbo

    prior = test_output  # [1, h, w, 1]

    projs_prior_full_view = projectors[0].forward_project(prior.transpose(1, 3).squeeze(1))
    fbp_prior_full_view = projectors[0].backward_project(projs_prior_full_view)

    projs_prior_sparse_view = projectors[1].forward_project(prior.transpose(1, 3).squeeze(1))
    fbp_prior_sparse_view = projectors[1].backward_project(projs_prior_sparse_view)

    streak_prior = (fbp_prior_sparse_view - fbp_prior_full_view).unsqueeze(1).transpose(1, 3)
    fbp_prior_sparse_view = fbp_prior_sparse_view.unsqueeze(1).transpose(1, 3)

    corrected_image = fbp_recon - streak_prior

    return corrected_image, prior, train_projections.squeeze().unsqueeze(0)