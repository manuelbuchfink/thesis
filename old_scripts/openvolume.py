import h5py

#read hdf5 image

#image = h5py.File('/lgrp/edu-2024-1-bsc-buchfiml/shapes_sparse_view/rand_1024projs_7.h5', 'r')   
image = h5py.File('/lgrp/edu-2024-1-bsc-buchfiml/rand_1024projs_7_corrected_with_128_projections.h5', 'w')   

print(list(image.keys()))
#image = image['Volume']
image['Volume'] = image['VolumeCorrected']
#image = image['VolumeCorrected']
print(list(image.keys()))
# for im in image:
#     print(im)