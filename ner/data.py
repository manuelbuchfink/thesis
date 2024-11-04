import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from utils import get_config, save_image

from skimage.feature import canny
from skimage.filters import sobel, gaussian
from ct_2d_projector import FanBeam2DProjector

def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))

def create_grid(h, w):
    h = int(h)
    w = int(w)
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def bounding_box_2D(img):   # function to compute minimal bounding box for one slice
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    if not np.where(rows)[0].any()  or not np.where(cols)[0].any():
        rmin = rmax = cmin = cmax = 0
    else:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

class ImageDataset_2D_sparsify(Dataset):
    def __init__(self, img_path, parser):

        #read hdf5 image
        image = h5py.File(img_path, 'r')           # list(image.keys()) = ['Tiles'], ['Volume']
        image = image['Volume']                    # (512,512,512) = [depth, height, width]


        self.img_dim = (image.shape[2], image.shape[1])
        num_slices = image.shape[0]

        self.slices = [None] * num_slices

        for i in range(num_slices):

            #split image into N evenly sized chunks
            self.slices[i] = image[i,:,:]           # (512,512) = [h, w]

            # Interpolate image to predefined size in case of smaller img size
            self.slices[i] = cv2.resize(self.slices[i], self.img_dim[::-1], interpolation=cv2.INTER_LINEAR)

            # Scaling normalization -> [0, 1]
            self.slices[i] = self.slices[i] / np.max(self.slices[i])
            self.slices[i] = np.nan_to_num(self.slices[i])

            self.slices[i] = torch.tensor(self.slices[i], dtype=torch.float32)[:, :, None] # [h, w, 1]

    def __getitem__(self, idx):
        return  self.slices[idx]              #return data tuple

    def __len__(self):
        return len(self.slices) # iterations

class ImageDataset_2D_hdf5_canny(Dataset):
    def __init__(self, img_path, parser):
        #read hdf5 image
        image = h5py.File(img_path, 'r')           # list(image.keys()) = ['Tiles'], ['Volume']
        image = image['Volume']                    # (512,512,512) = [depth, height, width]

        opts = parser.parse_args()
        config = get_config(opts.config)

        print(f"vol shape {image}")
        self.img_dim = (image.shape[2], image.shape[1])
        num_slices = image.shape[0]

        self.slices = [None] * num_slices
        self.grids = [None] * num_slices
        self.img_dims = [None] * num_slices

        for i in range(num_slices):

            #split image into N evenly sized chunks
            self.slices[i] = image[i,:,:]           # (512,512) = [h, w]

            # Interpolate image to predefined size in case of smaller img size
            self.slices[i] = cv2.resize(self.slices[i], self.img_dim[::-1], interpolation=cv2.INTER_LINEAR)

            # Scaling normalization -> [0, 1]
            self.slices[i] = self.slices[i] / np.max(self.slices[i])
            self.slices[i] = np.nan_to_num(self.slices[i])

            # #Canny Edge Detector to rid streak artifacts
            canny_image = self.slices[i]
            canny_image = canny(canny_image, sigma=4) # canny edge detector

            bounding_box = bounding_box_2D(canny_image) # compute bounding box

            # make sure that bounding box borders are even numbered
            rmin = bounding_box[0] - 1 if bounding_box[0] % 2 != 0 and bounding_box[0] > 0 else bounding_box[0]
            rmax = bounding_box[1] + 1 if bounding_box[1] % 2 != 0 else bounding_box[1]
            cmin = bounding_box[2] - 1 if bounding_box[2] % 2 != 0 and bounding_box[2] > 0 else bounding_box[2]
            cmax = bounding_box[3] + 1 if bounding_box[3] % 2 != 0 else bounding_box[3]

            self.slices[i] = torch.tensor(self.slices[i], dtype=torch.float32)[:, :, None] # [h, w, 1]
            self.slices[i] = self.slices[i][rmin:rmax, cmin:cmax, :]

            self.img_dims[i] =  (rmax, rmin, cmax, cmin)                                   #[h_new, w_new]
            img_dim = (self.img_dims[i][0] - self.img_dims[i][1], self.img_dims[i][2] - self.img_dims[i][3])

            self.grids[i] = (create_grid(*img_dim))

    def __getitem__(self, idx):
        return self.grids[idx].to(torch.float16) , self.slices[idx].to(torch.float16) , self.img_dims[idx]               #return data tuple

    def __len__(self):
        return len(self.slices) # iterations



class ImageDataset_3D_hdf5(Dataset):
    def __init__(self, img_path):

        #read hdf5 image
        image = h5py.File(img_path, 'r')           # list(image.keys()) = ['Tiles'], ['Volume']
        image = image['Volume']                    # (512,512,512) = [depth, height, width]

        image = torch.tensor(image, dtype=torch.float16)     # [B, C, H, W]

        self.img_dim = image.squeeze().shape

        # Scaling normalization
        image[image < 0] = 0
        image /= torch.max(image)                                       # [B, C, H, W], [0, 1]
        print(f"imageshape {self.img_dim}")
        self.img = image.unsqueeze(3)                 # [C, H, W, 1]
        display_tensor_stats(self.img)


    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img   # return data tuple

    def __len__(self):
        return 1                # iterations
