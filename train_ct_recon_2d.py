import os
import argparse
import shutil

import wandb
import torch
import torch.backends.cudnn as cudnn
import tensorboardX
from ct_2d_projector import FanBeam2DProjector
import numpy as np

from networks import Positional_Encoder, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader, save_image_2d
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import matplotlib
import gc
torch.cuda.empty_cache()
gc.collect()
import warnings
warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]


output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
.format(config['img_size'], config['num_proj_sparse_view_128'], config['model'], \
    config['net']['network_input_size'], config['net']['network_width'], \
    config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding']))
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
model = SIREN(config['net'])

model.cuda()
model.train()

optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
loss_fn = torch.nn.MSELoss().to("cuda")


# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])


ct_projector_full_view_512 = FanBeam2DProjector(config['img_size'], config['num_proj_full_view_512'])
ct_projector_sparse_view_128 = FanBeam2DProjector(config['img_size'], config['num_proj_sparse_view_128'])
ct_projector_sparse_view_64 = FanBeam2DProjector(config['img_size'], config['num_proj_sparse_view_64'])

wandb.init(
    # set the wandb project where this run will be logged
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
    print(grid.shape)
    image = image.cuda()  # [bs, x, y, 1], [0, 1]

    projs_512 = ct_projector_full_view_512.forward_project(image.transpose(1, 3).squeeze(1)) 
    fbp_recon_512 = ct_projector_full_view_512.backward_project(projs_512) 

    projs_128 = ct_projector_sparse_view_128.forward_project(image.transpose(1, 3).squeeze(1))  
    fbp_recon_128 = ct_projector_sparse_view_128.backward_project(projs_128) 

    projs_64 = ct_projector_sparse_view_64.forward_project(image.transpose(1, 3).squeeze(1))  
    fbp_recon_64 = ct_projector_sparse_view_64.backward_project(projs_64)  
    
    train_projs = projs_512[..., np.newaxis] #add dim for image save
    # Data loading
    test_data = (grid, image)
    train_data = (grid, projs_128)

    save_image_2d(test_data[1], os.path.join(image_directory, "test.png"))
    save_image_2d(train_projs, os.path.join(image_directory, "train.png"))

    fbp_recon_ssim = compare_ssim(fbp_recon_512.squeeze().cpu().numpy(), test_data[1].transpose(1,3).squeeze().cpu().numpy(), multichannel=True)  # [x, y, z] # treat the last dimension of the array as channels

    fbp_recon_512 = fbp_recon_512.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]
    fbp_recon_128 = fbp_recon_128.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]
    fbp_recon_64 = fbp_recon_64.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]

    fbp_recon_psnr = - 10 * torch.log10(loss_fn(fbp_recon_512, test_data[1]))

    save_image_2d(fbp_recon_512, os.path.join(image_directory, "fbprecon_512.png"))
    save_image_2d(fbp_recon_128, os.path.join(image_directory, "fbprecon_128.png"))
    save_image_2d(fbp_recon_64, os.path.join(image_directory, "fbprecon_64.png"))

    
     # Train model
    for iterations in range(max_iter):
        model.train()
        optim.zero_grad()

        train_embedding = encoder.embedding(train_data[0])  # [B, H, W, embedding*2]
        train_output = model(train_embedding) # [B, H, W, 3]
        train_projs = ct_projector_sparse_view_128.forward_project(train_output.transpose(1, 3).squeeze(1)).to("cuda")
        train_loss = (0.5 * loss_fn(train_projs.to("cuda"), train_data[1].to("cuda")).to("cuda")).to("cuda")
     
        train_loss.backward()
        optim.step()

        # Compute training psnr
        if (iterations + 1) % config['log_iter'] == 0:
            train_psnr = -10 * torch.log10(2 * train_loss).item()
            train_loss = train_loss.item()

            train_writer.add_scalar('train_loss', train_loss, iterations + 1)
            train_writer.add_scalar('train_psnr', train_psnr, iterations + 1)
            print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss, train_psnr))
            
        # Compute testing psnr
        if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
            model.eval()
            with torch.no_grad():
                test_embedding = encoder.embedding(test_data[0])
                test_output = model(test_embedding)

                test_loss = 0.5 * loss_fn(test_output.to("cuda"), test_data[1].to("cuda")).to("cuda")
                test_psnr = - 10 * torch.log10(2 * test_loss).item()
                test_loss = test_loss.item()

                test_ssim = compare_ssim(test_output.transpose(1,3).squeeze().cpu().numpy(), test_data[1].transpose(1,3).squeeze().cpu().numpy(), multichannel=True)

            train_writer.add_scalar('test_loss', test_loss, iterations + 1)
            train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
            save_image_2d(test_output, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(iterations + 1, test_psnr, test_ssim)))
            print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g}".format(iterations + 1, max_iter, test_loss, test_psnr, test_ssim))
            wandb.log({"acc": test_ssim, "loss": test_loss})
        # Save final model
                
        model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (iterations + 1))
        torch.save({'net': model.state_dict(), \
                    'enc': encoder.B, \
                    'opt': optim.state_dict(), \
                    }, model_name)
    '''
    Compute Streak artifacts from prior image
    '''    
    # get prior image once training is finished    
    prior = test_output 
    save_image_2d(prior, os.path.join(image_directory, "prior.png"))


    projs_prior_512 = ct_projector_full_view_512.forward_project(prior.transpose(1, 3).squeeze(1))  # [bs, n, w, z] -> [bs, n, h, w]
    fbp_prior_512 = ct_projector_full_view_512.backward_project(projs_prior_512)  # [bs, n, h, w] -> [bs, x, y, z]

    projs_prior_128 = ct_projector_sparse_view_128.forward_project(prior.transpose(1, 3).squeeze(1))  # [bs, n, w, z] -> [bs, n, h, w]
    fbp_prior_128 = ct_projector_sparse_view_128.backward_project(projs_prior_128)  # [bs, n, h, w] -> [bs, x, y, z]

    projs_prior_64 = ct_projector_sparse_view_64.forward_project(prior.transpose(1, 3).squeeze(1))  # [bs, n, w, z] -> [bs, n, h, w]
    fbp_prior_64 = ct_projector_sparse_view_64.backward_project(projs_prior_64)  # [bs, n, h, w] -> [bs, x, y, z]
    
    streak_prior_128 = fbp_prior_128 - fbp_prior_512
    streak_prior_64 = fbp_prior_64 - fbp_prior_512

    fbp_prior_512 = fbp_prior_512.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]
    fbp_prior_128 = fbp_prior_128.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]
    fbp_prior_64 = fbp_prior_64.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]

    save_image_2d(fbp_prior_512, os.path.join(image_directory, "fbp_prior_512.png"))
    save_image_2d(fbp_prior_128, os.path.join(image_directory, "fbp_prior_128.png"))
    save_image_2d(fbp_prior_64, os.path.join(image_directory, "fbp_prior_64.png"))


    streak_prior_64 = streak_prior_64.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]
    streak_prior_128 = streak_prior_128.unsqueeze(1).transpose(1, 3)  # [bs, z, x, y, 1]

    save_image_2d(streak_prior_64, os.path.join(image_directory, "streak_prior_64.png"))
    save_image_2d(streak_prior_128, os.path.join(image_directory, "streak_prior_128.png"))

    
    '''
    Compute Corrected image
    '''
    corrected_image_128 = fbp_recon_128 - streak_prior_128
    corrected_image_64 = fbp_recon_64 - streak_prior_64

    save_image_2d(corrected_image_64, os.path.join(image_directory, "corrected_image_64.png"))
    save_image_2d(corrected_image_128, os.path.join(image_directory, "corrected_image_128.png"))

    result_ssim_128 = compare_ssim(corrected_image_128.transpose(1,3).squeeze().cpu().numpy(), test_data[1].transpose(1,3).squeeze().cpu().numpy(), multichannel=True)
    result_ssim_64 = compare_ssim(corrected_image_64.transpose(1,3).squeeze().cpu().numpy(), test_data[1].transpose(1,3).squeeze().cpu().numpy(), multichannel=True)
    
    print(f'Result SSIM 128: {result_ssim_128}, Results SSIM 64: {result_ssim_64}')
    plt.gray()
    f, axarr = plt.subplots(4,1) 
    
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    recon512 = plt.imread(os.path.join(image_directory, 'fbprecon_512.png'))
    recon128 = plt.imread(os.path.join(image_directory, 'fbprecon_128.png'))
    recon64 = plt.imread(os.path.join(image_directory, 'fbprecon_64.png'))
    orig = plt.imread(os.path.join(image_directory, 'test.png'))
    
    axarr[0].imshow(orig)    
    axarr[1].imshow(recon512)    
    axarr[2].imshow(recon128)
    axarr[3].imshow(recon64)

    axarr[0].set_title('Test Image', fontstyle='italic')
    axarr[1].set_title('FBP Reconstruction - 512 Projections', fontstyle='italic')
    axarr[2].set_title('FBP Reconstruction - 128 Projections', fontstyle='italic')
    axarr[3].set_title('FBP Reconstruction - 64 Projections', fontstyle='italic')
    
    f1, axarr1 = plt.subplots(4,1) 
    
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    prior_plt = plt.imread(os.path.join(image_directory, 'prior.png'))
    fbp_prior_512_plt = plt.imread(os.path.join(image_directory, 'fbp_prior_512.png'))
    fbp_prior_128_plt = plt.imread(os.path.join(image_directory, 'fbp_prior_128.png'))
    fbp_prior_64_plt = plt.imread(os.path.join(image_directory, 'fbp_prior_64.png'))
    
    axarr1[0].imshow(prior_plt)    
    axarr1[1].imshow(fbp_prior_512_plt)    
    axarr1[2].imshow(fbp_prior_128_plt)
    axarr1[3].imshow(fbp_prior_64_plt)

    axarr1[0].set_title(f'Prior Image {model_name}', fontstyle='italic')
    axarr1[1].set_title('FBP Prior Reconstruction - 512 Projections', fontstyle='italic')
    axarr1[2].set_title('FBP Prior Reconstruction - 128 Projections', fontstyle='italic')
    axarr1[3].set_title('FBP Prior Reconstruction - 64 Projections', fontstyle='italic')



   
    f2, axarr2 = plt.subplots(4,1) 
    
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    streak_prior_64_plt = plt.imread(os.path.join(image_directory, 'streak_prior_64.png'))
    streak_prior_128_plt = plt.imread(os.path.join(image_directory, 'streak_prior_128.png'))
    corrected_image_64_plt = plt.imread(os.path.join(image_directory, 'corrected_image_64.png'))
    corrected_image_128_plt = plt.imread(os.path.join(image_directory, 'corrected_image_128.png'))

    
    axarr2[0].imshow(streak_prior_64_plt)    
    axarr2[1].imshow(streak_prior_128_plt)    
    axarr2[2].imshow(corrected_image_64_plt)
    axarr2[3].imshow(corrected_image_128_plt)

    axarr2[0].set_title('Streak Image - (64 Proj - 512 Proj)', fontstyle='italic')
    axarr2[1].set_title('Streak Image - (128 Proj - 512 Proj)', fontstyle='italic')
    axarr2[2].set_title('Corrected Image - 64 Projections', fontstyle='italic')
    axarr2[3].set_title('Corrected Image - 128 Projections', fontstyle='italic')
    plt.show()# what