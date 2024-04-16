
import numpy as np
data = np.load('/home/manuel/Documents/Bachelor Thesis/NeRP-main/data/ct_data/out_file.npz')
lst = data.files
np.set_printoptions(threshold=np.inf)
for item in lst:
    print(item.shape())
   # print(data[item])
#print(data[lst[0]][0][0])