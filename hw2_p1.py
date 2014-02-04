import numpy as np
from scipy import ndimage
from noise_estimation import noise_estimation

# Define the size of an image
n_row = 256
n_col = 256

# The number of images
n_image = 10

# Gray level
gray_level = 128

# Standard deviation of Gaussian noise
sigma = 2.0

# Array that stores images
img = np.zeros((n_image,n_row,n_col))

# Add noise to each image
for i in range(n_image):
    noise = np.random.normal(0,sigma,n_row*n_col).reshape(n_row,n_col)
    img[i,:,:] = np.ones((n_row,n_col))*gray_level + noise

# Problem 1
# Noise estimation of original images
noise_est = noise_estimation(img)

print "Noise Estimation: ",np.average(noise_est)

# Problem 2
# Filter the noisy image with Box filter
size_box = 3
filter_box = np.ones((size_box,size_box))*1./(size_box*size_box)

# images after filtering
img_box = np.zeros((n_image,n_row,n_col))
for i in range(n_image):
    img_box[i,:,:] = ndimage.filters.convolve(img[i,:,:],filter_box)
# Problem 2
# Noise estimation of image after box filtering
noise_box_est = noise_estimation(img_box)

print "Noise Estimation after box filtering: ",np.average(noise_box_est)

print np.average(noise_est)**2/size_box/size_box,np.average(noise_box_est)**2



