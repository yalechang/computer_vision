import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import os
from skimage import feature 
from skimage.transform import warp
from skimage.measure import ransac
import matplotlib
import cv2
import copy
import random
from scipy import interpolate

# Directories of the datasets
base_path = "/home/changyale/Dropbox/class_Spring_2014/computer_vision/"+\
        "Project/Project_2/"
imageset_1 = "DanaHallWay1/"
imageset_2 = "DanaHallWay2/"
imageset_3 = "DanaOffice/"

# Choose which dataset to use
flag_imageset = 1
img_a,img_b = (0,1)
if flag_imageset == 1:
    imageset = imageset_1
if flag_imageset == 2:
    imageset = imageset_2
if flag_imageset == 3:
    imageset = imageset_3

# List all filenames under given directory and choose .JPG files
file_names = os.listdir(base_path+imageset)
img_names = []
for i in file_names:
    if i[-4:] == '.JPG':
        img_names.append(base_path+imageset+i)

# Number of images
n_images = len(img_names)

# Read images into a array
# color images
img_color = []

# grayscale images
img = []
for i in range(len(img_names)):
    img_color.append(misc.imread(img_names[i]))
    img.append(misc.imread(img_names[i],flatten=1))

# Obtain the size of a single image
n_row,n_col = img[0].shape
img = np.array(img)
assert img.shape == (n_images,n_row,n_col)

# Extract Harris edge features from each image
feat_harris = []
for i in range(n_images):
    tmp = feature.corner_harris(img[i],method='k',k=0.05,\
            eps=1e-06,sigma=1.0)
    feat_harris.append(feature.peak_local_max(tmp,min_distance=10))

# Find image patches centered at corners and match them
radius = 5
ncc = np.zeros((feat_harris[img_a].shape[0],feat_harris[img_b].shape[0]))
for i in range(feat_harris[img_a].shape[0]):
    coord = feat_harris[img_a][i]
    if coord[0]+radius>n_row or coord[0]-radius<0 or coord[1]+\
                radius>n_col or coord[1]-radius<0:
        print "boarder of image",img_a,coord
    else:
        for j in range(feat_harris[img_b].shape[0]):
            tmp = feat_harris[img_b][j]
            if tmp[0]+radius>n_row or tmp[0]-radius<0 or tmp[1]+radius>n_col \
                    or tmp[1]-radius<0:
                print "boarder",img_b,tmp
            else:
                patch_a = img[img_a][coord[0]-radius:coord[0]+radius+1,\
                        coord[1]-radius:coord[1]+radius+1]
                patch_b = img[img_b][tmp[0]-radius:tmp[0]+radius+1,\
                        tmp[1]-radius:tmp[1]+radius+1]
                patch_a_normalized = patch_a/np.linalg.norm(patch_a)
                patch_b_normalized = patch_b/np.linalg.norm(patch_b)
                ncc[i,j] = np.sum(patch_a_normalized*patch_b_normalized)

# Combine two images
img_both = np.concatenate((img_color[img_a],img_color[img_b]),axis=0)
img_2 = copy.deepcopy(img_both)

# Note here for src, dst we use (x,y) coordinates in an image
src = []
dst = []
for i in range(ncc.shape[0]):
    for j in range(ncc.shape[1]):
        if ncc[i,j] == np.max(ncc[i,:]) and ncc[i,j]>0.98:
            p1 = (feat_harris[img_a][i][1],feat_harris[img_a][i][0])
            p2 = (feat_harris[img_b][j][1],feat_harris[img_b][j][0]+n_row)
            cv2.line(img_both,p1,p2,(0,0,255))
            src.append(p1)
            dst.append((feat_harris[img_b][j][1],feat_harris[img_b][j][0]))
src = np.array(src)
dst = np.array(dst)


# Apply Ransac
n_iter_max = 100
n_sample = 4
th = 3

cnt = np.zeros((src.shape[0],1))

for n_iter in range(n_iter_max):
    # Random sample points from src and dst
    idx = random.sample(range(src.shape[0]),n_sample)
    src_sample = src[idx,:]
    dst_sample = dst[idx,:]
    mtr_a = np.zeros((2*src_sample.shape[0],9))
    for i in range(src_sample.shape[0]):
        [x1,y1,x2,y2] = [src_sample[i,0],src_sample[i,1],dst_sample[i,0],\
                dst_sample[i,1]]
        mtr_a[i*2,:] = np.array([x1,y1,1,0,0,0,-x1*x2,-y1*x2,-x2])
        mtr_a[i*2+1,:] = np.array([0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2])
    u,s,v = np.linalg.svd(mtr_a.T.dot(mtr_a))
    h = u[:,-1].reshape(3,3)
    for i in list(set(range(src.shape[0]))-set(idx)):
        tmp = h.dot(np.hstack((src[i],1)).reshape(3,1))
        x2 = int(tmp[0,0]/tmp[2,0])
        y2 = int(tmp[1,0]/tmp[2,0])
        distance = np.sqrt((dst[i,0]-x2)**2+(dst[i,1]-y2)**2)
        if distance < th:
            cnt[idx,0] += 1


# selected corresponding pairs
idx_sel = []
for i in range(src.shape[0]):
    if cnt[i,0]>5:
        p1 = (src[i,0],src[i,1])
        p2 = (dst[i,0],dst[i,1]+n_row)
        cv2.line(img_2,p1,p2,(0,0,255))
        idx_sel.append(i)

# Use selected pairs to estimate the Homography
src_sel = src[idx_sel,:]
dst_sel = dst[idx_sel,:]
mtr_a = np.zeros((2*len(idx_sel),9))
for i in range(src_sel.shape[0]):
    [x1,y1,x2,y2] = [src_sel[i,0],src_sel[i,1],dst_sel[i,0],dst_sel[i,1]]
    mtr_a[i*2,:] = np.array([x1,y1,1,0,0,0,-x1*x2,-y1*x2,-x2])
    mtr_a[i*2+1,:] = np.array([0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2])
u,s,v = np.linalg.svd(mtr_a.T.dot(mtr_a))
h = u[:,-1].reshape(3,3)
h_inv = np.linalg.inv(h)

# Apply wrap
tmp = warp(img_color[img_b],h)
plt.figure(0)
plt.imshow(tmp)

print cnt
plt.figure(1)
plt.imshow(img_both)
plt.figure(2)
plt.imshow(img_2)
plt.show()


