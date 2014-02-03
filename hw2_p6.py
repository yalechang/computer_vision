import numpy as np
from scipy import ndimage

# Image size
n_row = 8
n_col = 8

img = np.zeros((n_row,n_col))

for i in range(n_row):
    for j in range(n_col):
        img[i,j] = abs(i-j)

img_median = ndimage.filters.median_filter(img,size=(3,3))


