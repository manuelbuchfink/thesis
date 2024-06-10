import os
import yaml

from torch.utils.data import DataLoader
import torchvision.utils as vutils

from data import ImageDataset_2D


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


def get_train_loader(img_path, img_dim, train, batch_size, num_workers, sampler, pin_memory):    

    dataset = ImageDataset_2D(img_path, img_dim)
    
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        drop_last=train, 
                        num_workers=num_workers,
                        sampler=sampler,
                        pin_memory=pin_memory
                        )
    return loader

def get_data_loader(img_path, img_dim, train, batch_size):    

    dataset = ImageDataset_2D(img_path, img_dim)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)
    return loader

def save_image_2d(tensor, file_name):
    '''
    tensor: [1, h, w, 1]
    '''
    tensor = tensor[0, ...].permute(2, 0, 1).cpu().data  # [1, h, w, 1] -> [1, h, w]
    image_grid = vutils.make_grid(tensor, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)
