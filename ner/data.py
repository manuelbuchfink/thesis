import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from utils import save_image
import torch.nn.functional as F

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

class ImageDataset_2D_hdf5(Dataset):
    def __init__(self, img_path, img_dim, num_slices):
        self.img_dim = (img_dim, img_dim) # [h, w]

        #read hdf5 image
        image = h5py.File(img_path, 'r')           # list(image.keys()) = ['Tiles'], ['Volume']
        image = image['Volume']                    # (512,512,512) = [depth, height, width]

        print(f"vol shape {image}")

        self.slices = [None] * num_slices
        self.grids = [None] * num_slices
        self.img_dims = [None] * num_slices

        for i in range(num_slices):

            #split image into N evenly sized chunks
            self.slices[i] = image[i,:,:]           # (512,512) = [h, w]
            #save_image(torch.tensor(self.slices[i], dtype=torch.float32), f"./untouched_im/image untouched, slice Nr. {i}.png")

            # Interpolate image to predefined size in case of smaller img size
            self.slices[i] = cv2.resize(self.slices[i], self.img_dim[::-1], interpolation=cv2.INTER_LINEAR)

            # Scaling normalization -> [0, 1]
            self.slices[i] = self.slices[i] / np.max(self.slices[i])
            self.slices[i] = np.nan_to_num(self.slices[i])

            '''

            compute the minimal bounding box encompassing the central object

            '''
            bounding_box = bounding_box_2D(self.slices[i]) # compute bounding box

            # make sure that bounding box borders are even numbered
            rmin = bounding_box[0] - 1 if bounding_box[0] % 2 != 0 and bounding_box[0] > 0 else bounding_box[0]
            rmax = bounding_box[1] + 1 if bounding_box[1] % 2 != 0 else bounding_box[1]
            cmin = bounding_box[2] - 1 if bounding_box[2] % 2 != 0 and bounding_box[2] > 0 else bounding_box[2]
            cmax = bounding_box[3] + 1 if bounding_box[3] % 2 != 0 else bounding_box[3]
            # rmin = cmin = 0
            # rmax = cmax = 512
            self.slices[i] = torch.tensor(self.slices[i], dtype=torch.float32)[:, :, None] # [h, w, 1]
            self.slices[i] = self.slices[i][rmin:rmax, cmin:cmax, :]

            self.img_dims[i] =  (rmax, rmin, cmax, cmin)                                   #[h_new, w_new]
            img_dim = (self.img_dims[i][0] - self.img_dims[i][1], self.img_dims[i][2] - self.img_dims[i][3])

            self.grids[i] = (create_grid(*img_dim))
            #display_tensor_stats(self.slices[i])


    def __getitem__(self, idx):
        return self.grids[idx], self.slices[idx], self.img_dims[idx]               #return data tuple

    def __len__(self):
        return len(self.slices) # iterations


class ImageDataset_3D_hdf5(Dataset):
    def __init__(self, img_path, img_dim):
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)

        #read hdf5 image
        image = h5py.File(img_path, 'r')           # list(image.keys()) = ['Tiles'], ['Volume']
        image = image['Volume']                    # (512,512,512) = [depth, height, width]

        print(f"vol shape {image}")

        # Crop slices in z dim
        center_idx = int(image.shape[0] / 2)
        num_slice = int(self.img_dim[0] / 2)
        image = image[center_idx-num_slice:center_idx+num_slice, :, :]
        im_size = image.shape


        # # Complete 3D input image as a squared x-y image
        # if not(im_size[1] == im_size[2]):
        #     zerp_padding = np.zeros([im_size[0], im_size[1], np.int((im_size[1]-im_size[2])/2)])
        #     image = np.concatenate([zerp_padding, image, zerp_padding], axis=-1)

        # Resize image in x-y
        print(f"pre im shape {image.shape}")
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
        print(f"pre mid shape {image.shape}")
        image = F.interpolate(image, size=(self.img_dim[1], self.img_dim[2]), mode='bilinear', align_corners=False)
        print(f"pre post shape {image.shape}")
        # Scaling normalization
        image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)


    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img               # return data tuple

    def __len__(self):
        return 1                            # iterations


class ImageDataset_3D_hdf5_direct(Dataset):
    def __init__(self, img_path, img_dim):
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)

        #read hdf5 image
        image = h5py.File(img_path, 'r')           # list(image.keys()) = ['Tiles'], ['Volume']
        image = image['Volume']                    # (512,512,512) = [depth, height, width]

        print(f"vol shape {image}")

        # Crop slices in z dim
        center_idx = int(image.shape[0] / 2)
        num_slice = int(self.img_dim[0] / 2)
        image = image[center_idx-num_slice:center_idx+num_slice, :, :]
        im_size = image.shape


        # # Complete 3D input image as a squared x-y image
        # if not(im_size[1] == im_size[2]):
        #     zerp_padding = np.zeros([im_size[0], im_size[1], np.int((im_size[1]-im_size[2])/2)])
        #     image = np.concatenate([zerp_padding, image, zerp_padding], axis=-1)

        # Resize image in x-y
        print(f"pre im shape {image.shape}")
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
        print(f"pre mid shape {image.shape}")
        #image = F.interpolate(image, size=(self.img_dim[1], self.img_dim[2]), mode='bilinear', align_corners=False)
        print(f"pre post shape {image.shape}")
        # Scaling normalization
        image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)


    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img               # return data tuple

    def __len__(self):
        return 1                            # iterations
