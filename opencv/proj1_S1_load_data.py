import numpy as np
import pylab
from scipy import misc
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import cv2

t0 = time.time()
# Directory of the images of a video
basepath = "/home/changyale/dataset/computer_vision/"
video_1 = "EnterExitCrossingPaths2cor/"

# Get the filenames of the images
file_names = os.listdir(basepath+video_1)
img_names = {}
for i in file_names:
    img_names[int(i[-8:-4])] = basepath+video_1+i

# Number of images
n_images = len(img_names)


# Read Images
img = []
for i in range(n_images):
    img.append(misc.imread(img_names[i],flatten=1))

# Get the size of one image
n_row,n_col = img[0].shape

# img.shape = (n_images,n_row,n_col)
img = np.array(img)
assert img.shape == (n_images,n_row,n_col)

# the difference between two neighboring images
th = 10
img_diff = np.zeros((n_images-2,n_row,n_col))
img_mask = np.zeros((n_images-2,n_row,n_col))

# 1D temporal derivative
for i in range(n_images-2):
    img_diff[i,:,:] = img[i+2,:,:]-img[i,:,:]
    img_mask[i,:,:] = abs(img_diff[i,:,:])>th

t1 = time.time()
print t1-t0

for i in range(n_images-2):
    print i,
    cv2.imshow('IMG',img_mask[i])
    cv2.waitKey(33)
cv2.destroyWindow('IMG')

