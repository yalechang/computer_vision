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

############################# Parameter Settings ###########################
# Choose image set
flag_imageset = 2

# Choose the pair of images in selected set
img_a,img_b = (0,1)

# Minimum distance for suppression in Harris Corner Detector
mindis_harris = 5

# The size of image patches used for computing normalized cross correlation
# in feature matching
radius = 5

# Threshold for normalized cross correlation in order to remove mismatches
# set this threshold to be 0.95 for cast, 0.98 for cone
th_ncc = 0.98

# RANSAC: Number of random sampling
n_iter_max = 1000
# RANSAC: number of samples(without replacement) in each random sampling
n_sample = 4
# RANSAC: threshold for distance between predicted coordinates by Homography
# and the paired coordinates
th_dis = 3
# RANSAC: threshold for number of matches, for each point in source image, if
# its number of matches is larger than th_cnt, then it should be inlier
# otherwise it should be outlier
th_cnt = 100
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
        if distance < th_dis:
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

# Determine output image size
src_bound = np.array([[1,1],[1,n_row],[n_col,n_row],[n_col,1]])
dst_bound = np.zeros(src_bound.shape)
for i in range(src_bound.shape[0]):
    tmp = h.dot(np.hstack((src_bound[i],1)).reshape(3,1))
    x2 = tmp[0,0]/tmp[2,0]
    y2 = tmp[1,0]/tmp[2,0]
    dst_bound[i,:] = np.array([x2,y2])

# Determine coordinates range of canvas
x_min = np.int(np.ceil(min(np.min(dst_bound[:,0]),1)))
x_max = np.int(np.floor(max(np.max(dst_bound[:,0]),n_col)))
y_min = np.int(np.ceil(min(np.min(dst_bound[:,1]),1)))
y_max = np.int(np.floor(max(np.max(dst_bound[:,1]),n_row)))
print x_min,x_max,y_min,y_max

img_mc = np.zeros((y_max-y_min+1,x_max-x_min+1,3))

bound_b = []
bound_a = []
for i in range(src_bound.shape[0]):
    bound_b.append(list(src_bound[i,:]))
    bound_a.append(list(dst_bound[i,:]))

# Begin mosaic
t0 = time()
print "Mosaic begins:"

xx,yy = np.meshgrid(range(x_min,x_max+1),range(y_min,y_max+1))

for x in range(img_mc.shape[1]):
    for y in range(img_mc.shape[0]):
        point = np.array([xx[y,x],yy[y,x]])
        # If point appears in img_b
        flag_b = point_in_poly(xx[y,x],yy[y,x],bound_b)
        # If point appears in img_a              
        flag_a = point_in_poly(xx[y,x],yy[y,x],bound_a)
        #print point,flag_a,flag_b
        # Case 1: pixel only appears in image_a
        if flag_a == True and flag_b == False:
            coord_b = point
            if coord_b[0]>n_col:
                coord_b[0] = n_col
            if coord_b[1]>n_row:
                coord_b[1] = n_row
            tmp = h_inv.dot(np.hstack((coord_b,1)).reshape(3,1))
            coord_a = np.array([tmp[0,0]/tmp[2,0],tmp[1,0]/tmp[2,0]])
            if coord_a[0]>n_col:
                coord_a[0] = n_col
            if coord_a[1]>n_row:
                coord_a[1] = n_row
            img_mc[y,x,:] = img_color[img_a][int(coord_a[1]-1),\
                    int(coord_a[0]-1),:]
        # Case 2: pixel only appears in image_b
        if flag_a == False and flag_b == True:
            coord_b = point
            if coord_b[0]>n_col:
                coord_b[0] = n_col
            if coord_b[1]>n_row:
                coord_b[1] = n_row
            img_mc[y,x,:] = img_color[img_b][coord_b[1]-1,coord_b[0]-1,:]
        # Case 3: pixel does not appear in image_a and image_b
        if flag_a == False and flag_b == False:
            pass
        # Case 4: pixel appear in both image_a and image_b, need blending
        if flag_a == True and flag_b == True:
            coord_b = point
            if coord_b[0]>n_col:
                coord_b[0] = n_col
            if coord_b[1]>n_row:
                coord_b[1] = n_row
            tmp = h_inv.dot(np.hstack((coord_b,1)).reshape(3,1))
            coord_a = np.array([tmp[0,0]/tmp[2,0],tmp[1,0]/tmp[2,0]])
            if coord_a[0]>n_col:
                coord_a[0] = n_col
            if coord_a[1]>n_row:
                coord_a[1] = n_row
            w_b = min([coord_b[0]-x_min,x_max-coord_b[0],coord_b[1]-y_min,\
                    y_max-coord_b[1]])
            w_a = min([coord_a[0]-x_min,x_max-coord_a[0],coord_a[1]-y_min,\
                    y_max-coord_a[1]])
            img_mc[y,x,:] = (w_a*img_color[img_b][int(coord_b[1]-1),\
                    int(coord_b[0]-1),:]+w_b*img_color[img_a][coord_a[1]-1,\
                    coord_a[0]-1,:])/(w_a+w_b)

print "Mosaic ends",time()-t0

#print cnt
plt.figure(1)
plt.imshow(img_1)
plt.figure(2)
plt.imshow(img_2)
plt.figure(3)
plt.imshow(img_mc.astype('uint8'))
plt.show()
