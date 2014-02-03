import numpy as np
from scipy import ndimage

# Image
img = np.array([[10,10,10,10,10,40,40,40,40,40]])

# Box filter
filter_box = 1./5*np.ones((1,5))

# Gaussian filter
filter_gaussian = 1./10*np.array([[1,2,4,2,1]])

# Filter with box filter
img_box = ndimage.filters.convolve(img,filter_box)

# Filter with Gaussian filter
img_gaussian = ndimage.filters.convolve(img,filter_gaussian)


