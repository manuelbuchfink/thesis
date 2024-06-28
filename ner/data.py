import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom as dicom
import h5py

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


class ImageDataset_2D(Dataset):

    def __init__(self, img_path, img_dim): 
        self.img_dim = (img_dim, img_dim) # [h, w]
        
        #read dicom image
        image = dicom.dcmread(img_path).pixel_array   
             
        # Interpolate image to predefined size in case of smaller img size
        image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR) 
        
        # Scaling normalization -> [0, 1]
        image = image / np.max(image)

        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None] # [h, w, 1]
        display_tensor_stats(self.img)
        
    def __getitem__(self, idx): 
        grid = create_grid(*self.img_dim)   # [h, w, 2]
        return grid, self.img               #return data tuple

    def __len__(self):
        return 1 # iterations
    
class ImageDataset_2D_hdf5(Dataset):

    def __init__(self, img_path, img_dim, img_slice): 
        self.img_dim = (img_dim, img_dim) # [h, w]
        
        #read hdf5 image
        image = h5py.File(img_path, 'r')   
        image = image['Volume'][img_slice,:,:]
             
        # Interpolate image to predefined size in case of smaller img size
        image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR) 
        
        # Scaling normalization -> [0, 1]
        image = image / np.max(image)

        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None] # [h, w, 1]
        display_tensor_stats(self.img)
        
    def __getitem__(self, idx): 
        grid = create_grid(*self.img_dim)   # [h, w, 2]
        return grid, self.img               #return data tuple

    def __len__(self):
        return 1 # iterations

class ImageDataset_2D_Slices(Dataset):

    def __init__(self, img_path, img_dim, batch_size): 
        self.img_dim = (img_dim, img_dim) # [h, w]
        chunk = img_dim/batch_size
        
        self.img_slice_dim = (chunk, img_dim)
        #read dicom image
        image = dicom.dcmread(img_path).pixel_array   
             
        # Interpolate image to predefined size in case of smaller img size
        image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR) 
        
        # Scaling normalization -> [0, 1]
        image = image / np.max(image)

        self.slices = [None] * batch_size
        self.grids = [None] * batch_size
        self.segments = [None] * batch_size
        for i in range(batch_size):
            chunk = img_dim/batch_size
            #split image into N evenly sized chunks
            self.slices[i] = torch.tensor(image[int(i*chunk):int((i+1)*chunk), :], dtype=torch.float32)[:, :, None] # [h/bs, w, 1]
            display_tensor_stats(self.slices[i])
            self.grids[i] = (create_grid(*self.img_slice_dim))
            self.segments[i] = (self.slices[i])  
        
    def __getitem__(self, idx):              
        return self.grids[idx], self.segments[idx]               #return data tuple

    def __len__(self):
        return len(self.slices) # iterations