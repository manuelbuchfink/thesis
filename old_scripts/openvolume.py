# import h5py
# from utils_simple import save_image

# import torchvision.utils as vutils
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T

# #image = h5py.File('/lgrp/edu-2024-1-bsc-buchfiml/shapes_sparse_view/rand_1024projs_7.h5', 'r')   
# image = h5py.File('/lgrp/edu-2024-1-bsc-buchfiml/rand_1024projs_7_corrected_with_128_projections_t0.9_skip_t_0.7_accuracy_new.hdf5', 'r')   

# print(list(image.keys()))

# image = image['Volume']

# slices = [None] * 512
# for i in range(512):           
            
#     #split image into N evenly sized chunks
#     slices[i] = image[i,:,:]           # (512,512) = [h, w]
#     save_image(torch.tensor(slices[i], dtype=torch.float32), f"./u_im_aftersaving/image from saved volume, slice Nr. {i}.png")

import h5py

import numpy

import matplotlib.pyplot as ply

import sys

import time

if len(sys.argv)<2:

   print ("Usage showHDF5 input.hdf5 proj/volume[0/1] slice_id")

   exit(1)

inFile= sys.argv[1]

tag="Image"

if sys.argv[2]=="0":

   tag="Image"

elif sys.argv[2]=="1":

   tag="Volume"

slice_id=int(sys.argv[3])

f_obj=h5py.File(inFile,'r')[tag][:,:,:]

dim=f_obj.shape

t1=numpy.arange(0,512)

# ply.figure(1)

# ply.imshow(f_obj[slice_id,:,:])

#ply.colorbar()

ply.figure(1)

ply.imshow(f_obj[:,slice_id,:])

ply.colorbar()
ply.show()
# print(image['GridSpacing'])
# for t in image['Type']:
#     print(t)

# #image['Volume'] = image['VolumeCorrected']
# #image = image['VolumeCorrected']
# print(list(image.keys()))
# # for im in image:
# #     print(im)

# import numpy
# import h5py
# import math,array,struct
# import matplotlib.pyplot as plt


# values=[]
# # read bin file
# Sum=0
# Sum2=0
# pixres=666
# pixres2=651

# if len(sys.argv)<2:
#     print("No enough args")
#     exit(1)
# scan_file=sys.argv[1]

# #gridSpacing=h5py.File('/import/scratch/tmp-ct-3/Ammar/PviotPointProject/R002189-Porsche_Full_01.hdf5','r')["GridSpacing"]
# #gridOrigin=h5py.File('/import/scratch/tmp-ct-3/Ammar/PviotPointProject/R002189-Porsche_Full_01.hdf5','r')["GridOrigin"]

# gridSpacing=[13.7e-06, 13.7e-06,13.7e-06]
# gridOrigin=[0, 0 ,0]


# file_out_hdf5 = h5py.File('/lgrp/edu-2024-1-bsc-buchfiml/rand_1024projs_7_corrected_with_128_projections_copy.h5', "w")




# a = array.array('h')
# a=numpy.fromfile(open(scan_file, 'rb'),dtype='float32')
# print(len(a))
# values=numpy.array(a,dtype="float32")


# print ("length of values:",len(values))
# noSlices=int(len(values)/(pixres*pixres2))
# print ("Number Of Slices:",noSlices)


# values=values[0:(noSlices*pixres*pixres2)]

# #data00=numpy.array(values)

# #data00[data00 < 0] = 0




# #values=values.reshape(noSlices,pixres,pixres2)
# values=values.reshape(pixres2,noSlices,pixres)
# #values=values.reshape(pixres,noSlices,pixres2)

# #b = numpy.transpose(values, (2, 1, 0))

# plt.figure(1)
# #plt.imshow(values[:,:,10])
# #plt.imshow(values[10,:,:])
# plt.imshow(values[:,100,:])
# plt.show()


# file_out_hdf5.create_dataset("Type", data=[86,111,108,117,109,101], shape=(6,1))
# file_out_hdf5.create_dataset("GridOrigin", data=gridOrigin, shape=(3,1))
# file_out_hdf5.create_dataset("GridSpacing", data=gridSpacing, shape=(3,1))
# volume_out = file_out_hdf5.create_dataset("Volume", dtype='float32', shape=(pixres2,noSlices,pixres,)) #t=np.array([rmin,cmin,zmin,rmax,cmax,zmax])

 
# for k in range (0,values.shape[1]):
#     volume_out[:,k,:]=values[:,k,:]



# file_out_hdf5.close()