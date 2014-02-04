import numpy as np
import pylab
from scipy import misc
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import cv2
from generate_filter import generate_filter

################################ Parameter Settings ##########################
# Dataset Selection Parameter
# (Required) Choose the video to use<1->EnterExitCrossingPaths2cor; 2->Office;
# 3->RedChair>
flag_video = 3

# Temporal Parameters
# (Required) Choose Temporal Filter<'diff'->0.5*[-1,0,1]; 'diff_gaussian'->1D
# filter of first derivative of Gaussian>
flag_t_filter = 'diff'
# (Optional) Choose kernel parameter for 'diff_gaussian'
t_sigma = 1.0

# Spatial Parameters
# (Required)Choose 2D Spatial Filter<'box'->box filter; 'gaussian'->2D 
# Gaussian filter>
flag_s_filter = 'box'
# (Optional) Choose kernel parameter for 'gaussian'
s_sigma = 1.0

t0 = time.time()
# Directory of the images of a video
basepath = "/home/changyale/dataset/computer_vision/"
video_1 = "EnterExitCrossingPaths2cor/"
video_2 = "Office/"
video_3 = "RedChair/"

if flag_video == 1:
    video = video_1
elif flag_video == 2:
    video = video_2
elif flag_video == 3:
    video = video_3
else:
    print "ERROR: Must Choose A Video"
# Get the filenames of the images
file_names = os.listdir(basepath+video)
img_names = {}
for i in file_names:
    img_names[int(i[-8:-4])] = basepath+video+i

# Number of images
n_images = len(img_names)

# Read Images
img = []
for i in img_names.keys():
    img.append(misc.imread(img_names[i],flatten=1))

# Get the size of one image
n_row,n_col = img[0].shape

# img.shape = (n_images,n_row,n_col)
img = np.array(img)
assert img.shape == (n_images,n_row,n_col)

# the difference between two neighboring images
th = 5
img_diff = np.zeros((n_images,n_row,n_col))
img_mask = np.zeros((n_images,n_row,n_col))

# 1D temporal derivative
if flag_t_filter == 'diff':
    w = 0.5*np.array([-1.,0.,1.])

img_diff[0,:,:] = w[0]*img[0,:,:]+w[2]*img[1,:,:]
img_diff[n_images-1,:,:] = w[0]*img[n_images-2,:,:]+w[2]*img[n_images-1,:,:]
for i in range(1,n_images-1):
    img_diff[i,:,:] = w[0]*img[i-1,:,:]+w[1]*img[i,:,:]+w[2]*img[i+1,:,:]

# Estimate noise for each image and set threshold for each image
img_noise = np.zeros((n_images,1))
for i in range(0,n_images-1):
    img_noise[i] = np.sum((img_diff[i+1,:,:]-img_diff[i,:,:])**2)/2/n_row/n_col
    img_mask[i,:,:] = abs(img_diff[i,:,:])>np.sqrt(img_noise[i])*3.0

t1 = time.time()
print t1-t0

for i in range(n_images):
    cv2.imshow('IMG',img_mask[i])
    cv2.waitKey(33)
cv2.destroyWindow('IMG')

