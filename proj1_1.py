import numpy as np
import pylab
from scipy import misc
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import cv2
from generate_filter import generate_filter
from scipy import ndimage

##################### Begin of Parameter Settings ##########################
# Dataset Selection Parameter
# (Required) Choose the video to use('enter'->EnterExitCrossingPaths2cor; 
# 'office'->Office; 'red'->RedChair)
flag_video = 'enter'

# Temporal Parameters
# (Required) Choose Temporal Filter('diff'->0.5*[-1,0,1]; 'diff_gaussian'->1D
# filter of first derivative of Gaussian)
flag_t_filter = 'diff'
# (Optional) Choose kernel parameter for 'diff_gaussian'
t_sigma = 0.1

# Spatial Parameters
# whether to apply spatial filtering('spatial','nonspatial')
flag_s = 'spatial'

# (Required)Choose 2D Spatial Filter('box'->box filter; 'gaussian'->2D 
# Gaussian filter)
flag_s_filter = 'gaussian'
# (Optional) Choose kernel parameter for 'gaussian'
s_sigma = 1000
filter_size = 5

# Whether to use adaptive threshold
flag_adaptive = 'adaptive'

# Default threshold
th = 5

# Display the whole video or selected images('video','images')
flag_display = 'video'

###################### END of Parameter Setting #############################

t0 = time.time()
# Directory of the images of a video
basepath = "/Users/changyale/dataset/computer_vision/"
video_1 = "EnterExitCrossingPaths2cor/"
video_2 = "Office/"
video_3 = "RedChair/"

if flag_video == 'enter':
    video = video_1
elif flag_video == 'office':
    video = video_2
elif flag_video == 'red':
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

# Read Images into a 3D array
img = []
for i in img_names.keys():
    img.append(misc.imread(img_names[i],flatten=1))

# Get the size of one image
n_row,n_col = img[0].shape

# img.shape = (n_images,n_row,n_col)
img = np.array(img)
assert img.shape == (n_images,n_row,n_col)

# Apply 2D spatial filtering before computing temporal derivative
if flag_s == 'spatial':
    filter_box = generate_filter('box',(filter_size,filter_size))
    filter_gaussian = generate_filter('gaussian',(filter_size,filter_size),\
            sigma=s_sigma)

    for i in range(n_images):
        img[i,:,:] = ndimage.filters.convolve(img[i,:,:],filter_box,\
                mode='constant') 

# Derivative images, which can be  obtained by applying temporal operator on 
# the original images
img_diff = np.zeros((n_images,n_row,n_col))

# Mask images, which can be obtained by applying threshold on the derivative
# images
img_mask = np.zeros((n_images,n_row,n_col))

# 1D temporal derivative
if flag_t_filter == 'diff':
    w = 0.5*np.array([-1.,0.,1.])
elif flag_t_filter == 'diff_gaussian':
    w = generate_filter('diff_gaussian',(1,5),sigma=t_sigma)

for i in range(n_row):
    for j in range(n_col):
        img_diff[:,i,j] = ndimage.filters.convolve(img[:,i,j],w,\
                mode='constant')

# Estimate noise for each image and set threshold for each image
img_noise = np.zeros((n_images,1))
for i in range(0,n_images-1):
    img_noise[i] = np.sum((img_diff[i+1,:,:]-img_diff[i,:,:])**2)/2/n_row/n_col
    if flag_adaptive == 'adaptive':
        th = np.sqrt(img_noise[i])*3.0
    img_mask[i,:,:] = abs(img_diff[i,:,:])>th
    
t1 = time.time()
print t1-t0

if flag_display == 'video':
    # Show the video
    for i in range(n_images):
        cv2.imshow('VIDEO',img[i,:,:]*img_mask[i,:,:])
        cv2.waitKey(33)
    cv2.destroyWindow('VIDEO')

if flag_display == 'images':
    # Show selected images
    for i in [100,200,300,400]:
        tmp = img[i]*img_mask[i]
        tmp_name = './experiment/img_'+flag_video+flag_t_filter+\
                str(t_sigma)+flag_s+flag_s_filter+str(s_sigma)+\
                str(filter_size)+flag_adaptive+str(i)+'.png'
        misc.imsave(tmp_name,tmp)
        cv2.imshow('IMAGES',tmp)
        cv2.waitKey(0)
    cv2.destroyWindow('IMAGES')


