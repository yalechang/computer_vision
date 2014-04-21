import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import os
from skimage import feature 
from skimage.measure import ransac
import matplotlib
import cv2
import copy
import random
from point_in_poly import point_in_poly
from time import time
from python.computer_vision.get_subwindow import get_subwindow

t0 = time()
############################# Parameter Settings ###########################
# Choose image set
flag_imageset = 1

# Choose the pair of images in selected set
img_a,img_b = (0,1)

# Minimum distance for suppression in Harris Corner Detector
mindis_harris = 5

# The size of image patches used for computing normalized cross correlation
# in feature matching
radius = 5

# Threshold for normalized cross correlation in order to remove mismatches
# set this threshold to be 0.95 for cast, 0.98 for cone
if flag_imageset == 1:
    th_ncc = 0.95
elif flag_imageset == 2:
    th_ncc = 0.98
else:
    pass

# RANSAC: Number of random sampling
n_iter_max = 100
# RANSAC: number of samples(without replacement) in each random sampling
n_sample = 8
# RANSAC: threshold for distance between predicted coordinates by Homography
# and the paired coordinates
th_dis = 1e-3
# RANSAC: threshold for number of matches, for each point in source image, if
# its number of matches is larger than th_cnt, then it should be inlier
# otherwise it should be outlier
th_cnt = 3
#############################################################################

# Directory of the dataset
base_path = "/home/changyale/dataset/computer_vision/proje/"
if flag_imageset == 1:
    img_names = [base_path+"cast-left.jpg",base_path+"cast-right.jpg"]
if flag_imageset == 2:
    img_names = [base_path+"Cones_im2.jpg",base_path+"Cones_im6.jpg"]

# Color images
img_color = []

# Grayscale images
img = []

for i in range(len(img_names)):
    img_color.append(misc.imread(img_names[i]))
    img.append(misc.imread(img_names[i],flatten=1))

n_images = len(img_names)
n_row,n_col = img[0].shape
img = np.array(img)

# Extract Harris edge features from each grayscale image
feat_harris = []
for i in range(n_images):
    # Harris corner detector
    tmp = feature.corner_harris(img[i],method='k',k=0.05,eps=1e-06,sigma=1.0)
    # Only keep the local maximum using nonmax suppression
    feat_harris.append(feature.peak_local_max(tmp,min_distance=mindis_harris))

# Compute normalized cross correlation between patches centered at corners of
# img_a and img_b
# ncc[i,j] measure the normalized cross correlation between i-th corner in
# image_a and j-th corner in image_b
ncc = np.zeros((feat_harris[img_a].shape[0],feat_harris[img_b].shape[0]))
for i in range(feat_harris[img_a].shape[0]):
    # coordinates in the order of (row_number,column_number) of i-th corner
    # in image_a 
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

# Stack the colored version(not grayscale) of img_a and img_b vertically
img_1 = np.concatenate((img_color[img_a],img_color[img_b]),axis=0)
img_2 = copy.deepcopy(img_1)

########################## Begin of RANSAC ##################################
# Remove outliers of corner pairs by running RANSAC
# src and dst store the coordinates of refined corner pairs in the source image
# img_a and destination image img_b respectively.
# Note here for src, dst we use (x,y) coordinates in an image, which is
# in contrary order compared to (row_number,column_number) returned by Harris
# corner detector
src = []
dst = []
for i in range(ncc.shape[0]):
    for j in range(ncc.shape[1]):
        if ncc[i,j] == np.max(ncc[i,:]) and ncc[i,j]>th_ncc:
            p1 = (feat_harris[img_a][i][1],feat_harris[img_a][i][0])
            p2 = (feat_harris[img_b][j][1],feat_harris[img_b][j][0])
            src.append(p1)
            dst.append(p2)
            # Draw correspondence pairs before RANSAC
            tmp = (p2[0],p2[1]+n_row)
            cv2.line(img_1,p1,tmp,(0,0,255))

# Transform src and dst from lists into array
src = np.array(src)
dst = np.array(dst)

# cnt stored the number of right matches for each corners in img_a in RANSAC
# if cnt[i] is larger than certain threshold th_dis, then 
cnt = np.zeros((src.shape[0],1))
for n_iter in range(n_iter_max):
    # Random sample points from src and dst
    idx = random.sample(range(src.shape[0]),n_sample)
    src_sample = src[idx,:]
    dst_sample = dst[idx,:]
    mtr_a = np.zeros((src_sample.shape[0],9))
    assert src_sample.shape[0]>=8
    for i in range(src_sample.shape[0]):
        [x1,y1,x2,y2] = [src_sample[i,0],src_sample[i,1],dst_sample[i,0],\
                dst_sample[i,1]]
        mtr_a[i,:] = np.array([x1*x2,x1*y2,x1,y1*x2,y1*y2,y1,x2,y2,1])
    u,s,v = np.linalg.svd(mtr_a.T.dot(mtr_a))
    h = u[:,-1].reshape(3,3)
    for i in list(set(range(src.shape[0]))-set(idx)):
        tmp = np.hstack((dst[i],1)).dot(h.dot(np.hstack((src[i],1)).reshape(3,1)))
        if abs(tmp) < th_dis:
            cnt[idx,0] += 1

# selected corresponding pairs
idx_sel = []
for i in range(src.shape[0]):
    if cnt[i,0]>th_cnt:
        p1 = (src[i,0],src[i,1])
        tmp = (dst[i,0],dst[i,1]+n_row)
        # Draw corresponding pairs after RANSAC
        cv2.line(img_2,p1,tmp,(0,0,255))
        idx_sel.append(i)
############################## End of RANSAC ################################

# Use selected pairs to estimate Fundamental matrix
src_sel = src[idx_sel,:]
dst_sel = dst[idx_sel,:]
mtr_a = np.zeros((src_sample.shape[0],9))
assert src_sample.shape[0]>=8
for i in range(src_sample.shape[0]):
    [x1,y1,x2,y2] = [src_sample[i,0],src_sample[i,1],dst_sample[i,0],\
            dst_sample[i,1]]
    mtr_a[i,:] = np.array([x1*x2,x1*y2,x1,y1*x2,y1*y2,y1,x2,y2,1])
u,s,v = np.linalg.svd(mtr_a.T.dot(mtr_a))
h = u[:,-1].reshape(3,3)
h_inv = np.linalg.inv(h)

# Compute a dense disparity map using the Fundamental matrix to help reduce the
# search space. The output should be two images, one image with the vertical
# disparity component, and another image with the horizontal disparity
# component,scale the grayscale so the lowest disparity is 0 and the highest
# disparity is 255.
disparity_vertical = np.zeros(img[img_a].shape)
disparity_horizontal = np.zeros(img[img_a].shape)
for i1 in range(img[img_a].shape[0]):
    t1 = time()
    for j1 in range(img[img_a].shape[1]):
        patch_a = get_subwindow(img[img_a],[i1,j1],radius)
        patch_a_normalized = patch_a/np.linalg.norm(patch_a)
        a,b,c = h.dot(np.hstack((np.array([j1,i1]),1)))
        ncc = []
        idx_ncc = []
        for i2 in range(img[img_b].shape[0]):
            if a != 0:
                j2 = int((-c-b*(i2))/a)
                if j2>=0 and j2<img[img_b].shape[1]:
                    patch_b = get_subwindow(img[img_b],[i2,j2],radius)
                    patch_b_normalized = patch_b/np.linalg.norm(patch_b)
                    ncc.append(np.sum(patch_a_normalized*patch_b_normalized))
                    idx_ncc.append([i2,j2])
        i2_sel,j2_sel = idx_ncc[ncc.index(max(ncc))]
        disparity_vertical[i1,j1] = i2_sel-i1
        disparity_horizontal[i1,j1] = j2_sel-j1
    t2 = time()
    print i1,t2-t1

# Normalize disparity to [0,255]
tmp = copy.deepcopy(disparity_vertical)
disparity_norm_v = np.floor(tmp*1./(np.max(tmp)-np.min(tmp))*255.)
tmp = copy.deepcopy(disparity_horizontal)
disparity_norm_h = np.floor(tmp*1./(np.max(tmp)-np.min(tmp))*255.)

# Visualization
plt.figure(1)
plt.imshow(img_1)
plt.figure(2)
plt.imshow(img_2)
plt.figure(3)
plt.imshow(disparity_norm_v.astype('uint8'))
plt.figure(4)
plt.imshow(disparity_norm_h.astype('uint8'))
plt.show()

