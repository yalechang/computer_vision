import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import os
from skimage import feature 
from python.computer_vision.non_maximal_suppresion import \
        non_maximal_suppresion

base_path = "/home/changyale/Dropbox/class_Spring_2014/computer_vision/"+\
        "Project/Project_2/"
imageset_1 = "DanaHallWay1/"
imageset_2 = "DanaHallWay2/"
imageset_3 = "DanaOffice/"

flag_imageset = 1
if flag_imageset == 1:
    imageset = imageset_1
if flag_imageset == 2:
    imageset = imageset_2
if flag_imageset == 3:
    imageset = imageset_3

file_names = os.listdir(base_path+imageset)
img_names = []
for i in file_names:
    img_names.append(base_path+imageset+i)

# Number of images
n_images = len(img_names)

# Read images into a 3D array
img = []
for i in range(len(img_names)):
    img.append(misc.imread(img_names[i],flatten=1))

# Get the size of one image
n_row,n_col = img[0].shape
img = np.array(img)
assert img.shape == (n_images,n_row,n_col)

feat_harris = np.zeros(img.shape)
# Harris Edge Detector
for i in range(n_images):
    tmp_1 = feature.corner_harris(img[i],method='k',k=0.04,\
            eps=1e-06,sigma=0.7)
    tmp_2 = feature.peak_local_max(tmp_1,min_distance=2,indices=False)
    feat_harris[i] = img[i]*tmp_2
    plt.figure(i)
    plt.imshow(feat_harris[i],cmap=cm.Greys_r)
plt.show()

