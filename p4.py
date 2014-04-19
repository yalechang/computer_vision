import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import os
from skimage import feature 
import matplotlib
import cv2
import copy
import random
from time import time
from get_subwindow import get_subwindow
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

################## Parameter Settings ########################
# Size of patches around interesting points
sz_patch = (25-1)/2

# Number of clusters for visual words
# n_clusters = 10

# Directories of dataset
base_path = "/home/changyale/dataset/computer_vision/"
train_set = "CarTrainImages/"
test_set = "CarTestImages/"
ground_truth = "GroundTruth/"

# Load Training set
file_names = os.listdir(base_path+train_set)
img_names = []
for i in file_names:
    if i[-4:] == '.jpg':
        img_names.append(base_path+train_set+i)
img_names.sort()
img = []
for i in range(len(img_names)):
    img.append(misc.imread(img_names[i],flatten=1))

# Load Testing set
file_names_test = os.listdir(base_path+test_set)
img_names_test = []
for i in file_names_test:
    if i[-4:] == '.jpg':
        img_names_test.append(base_path+test_set+i)
img_names_test.sort()
img_test = []
for i in range(len(img_names_test)):
    img_test.append(misc.imread(img_names_test[i],flatten=1))

# Step(a): Use a Harris corner detector to collect interesting points in the
# training images
feat_harris = []
for i in range(len(img)):
    # Harris corner detector
    tmp = feature.corner_harris(img[i],method='k',k=0.05,eps=1e-6,sigma=1.0)
    # Only keep the local maximum using nonmax suppression
    feat_harris.append(feature.peak_local_max(tmp,min_distance=10))

# Step(b): At each interesting point, extract a fixed size image patch(for
# example, 25 x 25 pixels) and use the vector of raw pixel intensities as the
# "descriptor"
feat_desc = []
for i in range(len(feat_harris)):
    if feat_harris[i].shape[0]>0:
        for j in range(feat_harris[i].shape[0]):
            tmp = get_subwindow(img[i],list(feat_harris[i][j,:]),sz_patch)
            tmp = tmp.reshape(1,tmp.shape[0]*tmp.shape[1])
            feat_desc.append(tmp[0,:])
feat_desc = np.array(feat_desc)

# Step(c): Cluster the patches images into clusters using K-means. This step is
# meant to significantly reduce the number of possible visual words. The
# clusters that you find here constitute a "visual vocabulary".
n_clusters_range = range(10,11)
score = []

for n_clusters in n_clusters_range:
    clf = KMeans(n_clusters=n_clusters,init='random',n_init=10,n_jobs=-1)
    clf.fit(scale(feat_desc))
    score.append(clf.score(feat_desc))
    print n_clusters
plt.plot(n_clusters_range,score)
plt.show()

# Step(d): Having found the vocabulary, go back to each training example and
# assign their local patches to visual words in the vocabulary. An image patch
# is assigned to the visual word which is closest using Euclidean distance(SSD)

# Step(e): For each visual word occurence in the training examples, record the
# possible displacement vectors between it and the object center. Assume the
# object center is the center of the cropped training image. 
