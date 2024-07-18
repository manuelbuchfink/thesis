import os
import yaml
import numpy as np

import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim


def save_image(tensor, file_name):
    '''
    tensor: [1, h, w]
    '''
    image_grid = vutils.make_grid(tensor, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)
    
