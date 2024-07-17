import h5py
import numpy as np

with h5py.File('../../lgrp/edu-2024-1-bsc-buchfiml/shapes_sparse_view/rand_512projs_8.h5', 'r') as f:
    print(list(f.keys()))

    volume = f['Volume']
    for v in volume:
        print(v.shape)
    print(volume)
  

