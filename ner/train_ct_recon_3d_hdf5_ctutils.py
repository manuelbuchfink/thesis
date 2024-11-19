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

from networks import Positional_Encoder_3D, FFN_3D
from ct_3d_projector_ctutils import ConeBeam3DProjector
from utils import get_config, prepare_sub_folder, get_data_loader_hdf5, save_volume, save_image_2d
from data import ImageDataset_3D_hdf5

import torch # pylint: disable=import-error
import torch.backends.cudnn as cudnn # pylint: disable=import-error

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
    slice_nr = 86
    row_nr = 86
    column_start = 0
    column_end = 170

    n = config['down_sample_factor']
    image = image.squeeze()[::n,::n,::n].unsqueeze(0).unsqueeze(4).cuda()    # [1, h, w, d, 1], value range = [0, 1]
    grid = grid.squeeze()[::n,::n,::n].unsqueeze(0).cuda()                   # [1, h, w, d, 3], value range = [0, 1]

    output_subfolder = config['data']
    model_name = os.path.join(output_folder, f"{output_subfolder}/img{list(image.squeeze().shape)}_proj{config['num_proj_sparse_view']}_{config['lr']}_scale{config['encoder']['scale']}_size{config['encoder']['embedding_size']}")
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    print(model_name)
    line_image = profile_line(image.squeeze().float().squeeze().cpu().detach().numpy()[slice_nr,:,:], (row_nr,column_start), (row_nr,column_end), 1)

    ct_projector_sparse_view = ConeBeam3DProjector(image.squeeze().shape, config['cb_para'])
    projections = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))    # [1, h, w, 1] -> [1, 1, w, h] -> ([1, w, h]) -> [1, num_proj_sparse_view, original_image_size]
    fbp_recon= ct_projector_sparse_view.backward_project(projections)                           # ([1, num_proj_sparse_view, original_image_size]) -> [1, w, h]

    fbp_recon = fbp_recon.unsqueeze(1).transpose(1, 4)                                          # [1, h, w, 1]
    fbp_recon = torch.tensor(fbp_recon, dtype=torch.float16)                                    # [B, C, H, W]
    fbp_volume = torch.tensor(fbp_recon, dtype=torch.float16)
    line_proj = profile_line(projections.squeeze().float().squeeze().cpu().detach().numpy()[int(config['num_proj_sparse_view']/2),:,:], (row_nr,column_start), (row_nr,column_end), 1)
    line_fbp = profile_line(fbp_recon.squeeze().float().squeeze().cpu().detach().numpy()[slice_nr,:,:], (row_nr,column_start), (row_nr,column_end), 1)
    plt.plot(line_image)
    #plt.plot(line_proj)
    plt.plot(line_fbp)
    #plt.legend(['proj'], loc='upper left')
    plt.legend(['image', 'fbp'], loc='upper left')

    plt.xlabel("column")
    plt.ylabel("intensity")
    plt.title(f'line profile slice {slice_nr}, row {row_nr}, columns [{column_start}, {column_end}], small')

    plt.savefig(os.path.join(image_directory,f'line_profile_slice_{slice_nr}_row_{row_nr}_columns_[{column_start}, {column_end}]_small.png'), bbox_inches='tight')
    plt.show()
    plt.clf()

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
            train_projections = ct_projector_sparse_view.forward_project(train_output.transpose(1, 4).squeeze(1)).to("cuda")      # evaluate by forward projecting
            #train_loss = (0.5 * loss_fn(train_output.to("cuda"), fbp_recon.to("cuda")))     # compare forward projected grid with sparse view projection
            train_loss = (0.5 * loss_fn(train_projections.to("cuda"), projections.to("cuda")))     # compare forward projected grid with sparse view projection

        scaler.scale(fbp_recon)
        scaler.scale(train_projections)
        scaler.scale(train_loss).backward()
        scaler.step(optim)
        scaler.update()

        if (iterations + 1) % config['val_iter'] == 0: # compute metrics
            slice_nr = 87
            row_nr = 87
            column_start = 0
            column_end = 170
            save_image_2d(train_output[:,87,:,:,:].float(), os.path.join(image_directory, f"test_slice_{iterations + 1}.png"))
            line_original = profile_line(image.float().squeeze().cpu().detach().numpy()[slice_nr,:,:], (row_nr,column_start), (row_nr,column_end), 1)
            line_prior = profile_line(train_output.float().squeeze().cpu().detach().numpy()[slice_nr,:,:], (row_nr,column_start), (row_nr,column_end), 1)
            line_fbp = profile_line(fbp_recon.squeeze().float().squeeze().cpu().detach().numpy()[slice_nr,:,:], (row_nr,column_start), (row_nr,column_end), 1)

            plt.plot(line_original)
            plt.plot(line_prior, linewidth=2, linestyle=(0, (1, 1)))
            plt.plot(line_fbp)
            plt.legend(['ground truth', 'train_output', 'fbp'], loc='upper left')

            plt.xlabel("column")
            plt.ylabel("intensity")
            plt.title(f'line profile slice {slice_nr}, row {row_nr}, columns [{column_start}, {column_end}], iterations {iterations + 1}')

            plt.savefig(os.path.join(image_directory,f'line_profile_slice_{slice_nr}_row_{row_nr}_columns_[{column_start}, {column_end}]_iterations_{iterations + 1}.png'), bbox_inches='tight')
            plt.show()
            plt.clf()
            save_image_2d(train_output[:,86,:,:,:].float(), os.path.join(image_directory, f"test_slice_{iterations + 1}.png"))
            model.eval()
            with torch.no_grad():
                fbp_prior = ct_projector_sparse_view.backward_project(train_projections).unsqueeze(1).transpose(1, 4)
                test_ssim = compare_ssim(fbp_prior.transpose(1,4).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,4).squeeze().cpu().detach().numpy(), multichannel=True, data_range=1.0)
                test_mse = mse(fbp_prior.transpose(1,4).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,4).squeeze().cpu().detach().numpy())
                test_psnr = psnr(fbp_prior.transpose(1,4).squeeze().cpu().detach().numpy(), fbp_recon.transpose(1,4).squeeze().cpu().detach().numpy(), data_range=1.0)

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

    ct_projector_full_view = ConeBeam3DProjector(fbp_volume.squeeze().shape, config['cb_para_full'])
    ct_projector_sparse_view = ConeBeam3DProjector(fbp_volume.squeeze().shape, config['cb_para'])

    projs_prior_full_view = ct_projector_full_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
    fbp_prior_full_view = ct_projector_full_view.backward_project(projs_prior_full_view)
    fbp_prior_full_view = fbp_prior_full_view.unsqueeze(1).transpose(1, 4)

    projs_prior_sparse_view = ct_projector_sparse_view.forward_project(prior_volume.transpose(1, 4).squeeze(1))
    fbp_prior_sparse_view = ct_projector_sparse_view.backward_project(projs_prior_sparse_view)
    fbp_prior_sparse_view = fbp_prior_sparse_view.unsqueeze(1).transpose(1, 4)

    streak_volume = (fbp_prior_sparse_view - fbp_prior_full_view)
    corrected_volume = (fbp_volume.unsqueeze(4) - streak_volume)
    difference_volume = (corrected_volume.squeeze().cuda() - image.squeeze().cuda()).cuda()
    streak_original = (image.squeeze().cuda() - fbp_volume.squeeze().cuda()).cuda()
    streak_difference = streak_original.squeeze() - streak_volume.squeeze().cuda()


    projs_fbp_full_view = ct_projector_full_view.forward_project(image.transpose(1, 4).squeeze(1))      #just performs amazing right now because the availalbe image input has full projections!!!
    fbp_full_view = ct_projector_full_view.backward_project(projs_fbp_full_view)                        #just performs amazing right now because the availalbe image input has full projections!!!
    fbp_full_view = fbp_full_view.unsqueeze(1).transpose(1, 4)                                          #just performs amazing right now because the availalbe image input has full projections!!!

    projs_fbp_sparse_view = ct_projector_sparse_view.forward_project(image.transpose(1, 4).squeeze(1))  #just performs amazing right now because the availalbe image input has full projections!!!
    fbp_sparse_view = ct_projector_sparse_view.backward_project(projs_fbp_sparse_view)                  #just performs amazing right now because the availalbe image input has full projections!!!
    fbp_sparse_view = fbp_sparse_view.unsqueeze(1).transpose(1, 4)                                      #just performs amazing right now because the availalbe image input has full projections!!!

    fbp_streak_volume = (fbp_sparse_view - fbp_full_view)                                               #just performs amazing right now because the availalbe image input has full projections!!!
    fbp_corrected_volume = (fbp_volume.unsqueeze(4) - fbp_streak_volume)                                #just performs amazing right now because the availalbe image input has full projections!!!

    streak_difference_train_minus_direct = streak_volume - fbp_streak_volume
    prior_minus_image = prior_volume - image


    orient1_slice_corrdiff = prior_volume.squeeze().unsqueeze(0).unsqueeze(4).transpose(1, 4) - image.squeeze().unsqueeze(0).unsqueeze(4).transpose(1, 4)
    orient2_slice_corrdiff = prior_volume.squeeze().unsqueeze(0).unsqueeze(4).transpose(3, 4) - image.squeeze().unsqueeze(0).unsqueeze(4).transpose(3, 4)

    slice_nr = 87
    row_nr = 87
    column_start = 0
    column_end = 170
    save_image_2d(fbp_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"FBP_volume.png"))
    save_image_2d(corrected_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"corrected_volume.png"))
    save_image_2d(prior_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"prior_volume.png"))
    save_image_2d(streak_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"streak_volume_prior_difference.png"))
    save_image_2d(fbp_prior_full_view.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"fbp_prior_full_view.png"))
    save_image_2d(fbp_prior_sparse_view.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"fbp_prior_sparse_view.png"))
    save_image_2d(difference_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"difference_volume_corrected_minus_image.png"))
    save_image_2d(streak_original.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"streak_original_image_minus_fbp.png"))
    save_image_2d(streak_difference.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"streak_difference_original_minus_prior.png"))
    save_image_2d(fbp_corrected_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"fbp_direct_corrected.png")) #just performs amazing right now because the availalbe image input has full projections!!!
    save_image_2d(fbp_streak_volume.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"streak_fbp_direct.png"))
    save_image_2d(streak_difference_train_minus_direct.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"streak_difference_train_minus_direct.png"))
    save_image_2d(prior_minus_image.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"prior_minus_image.png"))
    save_image_2d(orient1_slice_corrdiff.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"prior_minus_image_skewed.png"))
    save_image_2d(orient2_slice_corrdiff.squeeze().float().unsqueeze(0).unsqueeze(4)[:,slice_nr,:,:,:], os.path.join(image_directory, f"prior_minus_image_skewed2.png"))

    line_original = profile_line(image.float().squeeze().cpu().numpy()[slice_nr,:,:], (row_nr,column_start), (row_nr,column_end), 1)
    line_prior = profile_line(prior_volume.float().squeeze().cpu().numpy()[slice_nr,:,:], (row_nr,column_start), (row_nr,column_end), 1)
    # print(f"ground truth line slice {slice_nr}, row {row_nr}, columns ({column_start, column_end}): {line_original}")
    # print(f"prior line slice {slice_nr}, row {row_nr}, columns ({column_start, column_end}): {line_prior}")
    # print(f"difference line slice {slice_nr}, row {row_nr}, columns ({column_start, column_end}): {line_original-line_prior}")
    plt.plot(line_original)
    plt.plot(line_prior, linewidth=2, linestyle=(0, (1, 1)))

    plt.legend(['ground truth', 'train_output'], loc='upper left')

    plt.xlabel("column")
    plt.ylabel("intensity")
    plt.title(f'line profile slice {slice_nr}, row {row_nr}, columns [{column_start}, {column_end}], iterations {max_iter}')

    plt.savefig(os.path.join(image_directory,f'line_profile_slice_{slice_nr}_row_{row_nr}_columns_[{column_start}, {column_end}]_iterations_{max_iter}.png'), bbox_inches='tight')
    plt.show()

    difference_volume = difference_volume.squeeze().cpu().detach().numpy()
    streak_original = streak_original.squeeze().cpu().detach().numpy()
    streak_difference = streak_difference.squeeze().cpu().detach().numpy()
    fbp_prior_full_view = fbp_prior_full_view.squeeze().cpu().detach().numpy()
    fbp_prior_sparse_view = fbp_prior_sparse_view.squeeze().cpu().detach().numpy()
    fbp_volume = fbp_volume.squeeze().cpu().detach().numpy()
    prior_volume = prior_volume.squeeze().cuda().cpu().detach().numpy()
    streak_volume = streak_volume.squeeze().cuda().cpu().detach().numpy()
    corrected_volume = corrected_volume.squeeze().cpu().detach().numpy()
    image = image.squeeze().cpu().detach().numpy()

    test_mse = mse(image, corrected_volume)
    test_ssim = compare_ssim(image, corrected_volume, axis=-1, data_range=1.0)
    test_psnr = psnr(image, corrected_volume, data_range=1.0)

    print(f"FINAL SSIM: {test_ssim}, MSE: {test_mse}, PSNR: {test_psnr}")
    # save_volume(fbp_volume, image_directory, config, "fbp_volume")
    # save_volume(corrected_volume, image_directory, config, "corrected_volume")
    # save_volume(prior_volume, image_directory, config, "prior_volume")
    # save_volume(streak_volume, image_directory, config, "streak_volume")
    # save_volume(fbp_prior_full_view, image_directory, config, "fbp_prior_full_view")
    # save_volume(fbp_prior_sparse_view, image_directory, config, "fbp_prior_sparse_view")
    # save_volume(difference_volume, image_directory, config, "difference_volume")
    # save_volume(streak_original, image_directory, config, "difference_volume")
    # save_volume(streak_difference, image_directory, config, "streak_difference")
