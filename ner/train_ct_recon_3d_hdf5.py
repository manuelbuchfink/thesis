
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

import h5py # pylint: disable=import-error
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_image, save_image_2d

from networks import Positional_Encoder_3D, FFN_3D
from ct_3d_projector import ConeBeam3DProjector

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error
import torch.nn.functional as F # pylint: disable=import-error
import tensorboardX # pylint: disable=import-error
from skimage.metrics import structural_similarity as compare_ssim # pylint: disable=import-error
from data import ImageDataset_3D_hdf5

warnings.filterwarnings("ignore")
sys.path.append('zhome/buchfiml/miniconda3/envs/odl/lib/python3.11/site-packages')
sys.path.append(os.getcwd())

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--id', type=str, default='.', help="id slice")
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

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Setup data loader
print('Load volume: {}'.format(config['img_path']))
dataset = ImageDataset_3D_hdf5(config['img_path'], config['dataset_size'])
data_loader = get_data_loader_hdf5(dataset, batch_size=config['batch_size'])

for it, (grid, image) in enumerate(data_loader):

    # Input coordinates (h,w) grid and target image
    grid = grid.cuda()      # [1, h, w, d, 3], value range = [0, 1]
    image = image.cuda()    # [1, h, w, d, 1], value range = [0, 1]
    grid = grid[:,int(int(opts.id) * 128):int(((int(opts.id) + 1) * 128)),:,:]
    #image = image[:,int(int(opts.id) * 128):int(((int(opts.id) + 1) * 128)),:,:]
    print(f"id 0 {int(int(opts.id) * 128)} id 1 {int(((int(opts.id) + 1) * 128))}")
    print(grid.shape, image.shape)


    '''
    with available data
    '''
    image = torch.load(os.path.join(image_directory, f"fbp_volume.pt"))[int(int(opts.id) * 128):int(((int(opts.id) + 1) * 128)),:,:].cuda() # [128, 512, 512]
    image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
    image = F.interpolate(image, size=(config['dataset_size'][1], config['dataset_size'][2]), mode='bilinear', align_corners=False)
    image = image.unsqueeze(4)

    #ct_projector_full_view = ConeBeam3DProjector(config['img_size'], proj_size=config['proj_size'], num_proj=config['num_proj_full_view'])
    ct_projector_sparse_view = ConeBeam3DProjector(config['img_size'], proj_size=config['proj_size'], num_proj=config['num_proj_sparse_view'])
    #projectors = [ct_projector_full_view, ct_projector_sparse_view]

    projections = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
    fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

    fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4)                                          # [1, h, w, 1]

    # Setup input encoder:
    encoder = Positional_Encoder_3D(config['encoder'])

    # Setup model
    model = FFN_3D(config['net'])


    model.cuda()
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    loss_fn = torch.nn.MSELoss().cuda()

    train_embedding = encoder.embedding(grid)  # fourier feature embedding:  ([1, x, y, z, 3] * [3, embedding_size]) -> [1, z, x, y, embedding_size]

    # Train model
    for iterations in range(max_iter):

        model.train()
        optim.zero_grad()

        train_output = model(train_embedding)  # train model on grid: ([1, x, y, embedding_size]) > [1, x, y, 1]

        train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 4).squeeze(1)).to("cuda")      # evaluate by forward projecting
        #train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))                                    # compare forward projected grid with sparse view projection
        train_loss = (0.5 * loss_fn(train_output.to("cuda"), image.to("cuda")))
        train_loss.backward()
        optim.step()

        # Compute ssim
        if (iterations + 1) % config['val_iter'] == 0:

            model.eval()
            with torch.no_grad():
                fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 4)
                test_ssim = compare_ssim(fbp_prior.transpose(1,4).squeeze().cpu().numpy(), fbp_recon.transpose(1,4).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
                #test_ssim_direct = compare_ssim(train_projections.squeeze().cpu().detach().numpy(), projections.squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
            end = time.time()

            print("[Slice Nr. {} Iteration: {}/{}] | FBP SSIM: {:.4g} | Time Elapsed: {}".format(it + 1, iterations + 1, max_iter, test_ssim, (end - start) / 60))


            if (iterations + 1) == max_iter:
                torch.save(train_output, os.path.join(image_directory, f"prior_volume_{opts.id}.pt"))
                #torch.save(fbp_recon, os.path.join(image_directory, f"fbp_volume_{opts.id}.pt"))
#                 prior = train_output.cuda()  # [1, h, w, 1]

#                 projs_prior_full_view = projectors[0].forward_project(prior.transpose(1, 4).squeeze(1))
#                 fbp_prior_full_view = projectors[0].backward_project(projs_prior_full_view)

#                 projs_prior_sparse_view = projectors[1].forward_project(prior.transpose(1, 4).squeeze(1))
#                 fbp_prior_sparse_view = projectors[1].backward_project(projs_prior_sparse_view)

#                 streak_prior = (fbp_prior_sparse_view - fbp_prior_full_view).unsqueeze(1).transpose(1, 4)
#                 fbp_prior_sparse_view = fbp_prior_sparse_view.unsqueeze(1).transpose(1, 4)

#                 corrected_image = fbp_recon - streak_prior

#                 diff_ssim_recon = compare_ssim(fbp_recon.transpose(1,4).squeeze().cpu().detach().numpy(), image.transpose(1,4).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)
#                 diff_ssim_train = compare_ssim(corrected_image.transpose(1,4).squeeze().cpu().detach().numpy(), image.transpose(1,4).squeeze().cpu().numpy(), multichannel=True, data_range=1.0)

#                 print(f"Diff SSIM TRAIN = {diff_ssim_train}, Diff SSIM RECON = {diff_ssim_recon}")



#                 fbp_padded = F.pad(fbp_recon, (0,0, pads[2],pads[3], pads[0],pads[1]))
#                 prior_padded = F.pad(prior, (0,0, pads[2],pads[3], pads[0],pads[1]))
#                 image_padded = F.pad(image, (0,0, pads[2],pads[3], pads[0],pads[1]))

#                 train_projections = train_projections.squeeze().unsqueeze(0)
#                 train_pad = int((config['img_size'] - config['num_proj_sparse_view']) / 2)
#                 train_projections_padded = F.pad(train_projections, (0,0, train_pad,train_pad)).unsqueeze(3)

#                 output_image =  torch.cat(((train_projections_padded / torch.max(train_projections_padded)), fbp_padded, prior_padded,  corrected_image_padded), 2)
#                 save_image_2d(output_image, os.path.join(image_directory, f"outputs_slice_{it + 1}_iter_{iterations + 1}_SSIM_{diff_ssim_train}.png"))
#                 save_image_2d(fbp_padded, os.path.join(image_directory, f"fbp_slice_{it + 1}_iter_{iterations + 1}_SSIM_{diff_ssim_train}.png"))
#                 save_image_2d(prior_padded, os.path.join(image_directory, f"prior_slice_{it + 1}_iter_{iterations + 1}_SSIM_{diff_ssim_train}.png"))
#                 save_image_2d(corrected_image_padded, os.path.join(image_directory, f"corrected_slice_{it + 1}_iter_{iterations + 1}_SSIM_{diff_ssim_train}.png"))
#                 save_image_2d(streak_prior, os.path.join(image_directory, f"streak_slice_{it + 1}_iter_{iterations + 1}_SSIM_{diff_ssim_train}.png"))


#         total_its+=1

#     # Save current model
#     model_name = os.path.join(checkpoint_directory, 'temp_model.pt')
#     torch.save({'net': model.state_dict(), \
#                 'enc': encoder.B, \
#                 'opt': optim.state_dict(), \
#                 }, model_name)




# corrected_images = torch.cat(corrected_images, 0)
# sparse_images = torch.cat(sparse_images, 0).squeeze()
# print(f"total iterations: {total_its}")

# # save corrected slices in new hdf5 Volume
# corrected_image_path = os.path.join(image_directory, f"../{config['data'][:-3]}_corrected_with_{config['num_proj_sparse_view']}_projections.hdf5")
# print(f"saved to {config['data'][:-3]}_corrected_with_{config['num_proj_sparse_view']}_projections.hdf5")

# sparse_image_path = os.path.join(image_directory, f"../{config['data'][:-3]}_sparse_view_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")
# print(f"saved to {config['data'][:-3]}_sparse_with_{config['num_proj_sparse_view']}_projections_t{config['slice_skip_threshold']}_skip_t_{config['accuracy_goal']}_accuracy.hdf5")

# gridSpacing=[5.742e-05, 5.742e-05, 5.742e-05]
# gridOrigin=[0, 0 ,0]
# with h5py.File(corrected_image_path,'w') as hdf5:
#     hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
#     hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
#     hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
#     hdf5.create_dataset("Volume", data=np.asarray(corrected_image))

# with h5py.File(sparse_image_path,'w') as hdf5:
#     hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
#     hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
#     hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
#     hdf5.create_dataset("Volume", data=np.asarray(sparse_images))

# # image_fbp_direct = h5py.File(sparse_image_path, 'r')
# # image_fbp_direct = image_fbp_direct['Volume']
# corrected_volume = h5py.File(corrected_image_path, 'r')
# corrected_volume = corrected_volume['Volume']
# slices_sparse = [None] * int(config['img_size'][0])
# for i in range(int(config['img_size'][0])):

#     #split image into N evenly sized chunks
#     slices_sparse[i] = corrected_volume[i,:,:]           # (512,512) = [h, w]
#     save_image(torch.tensor(slices_sparse[i], dtype=torch.float32), f"./u_volume_corrected_after_saving/image from saved volume, slice Nr. {i}.png")
