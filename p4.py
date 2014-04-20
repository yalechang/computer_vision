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
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from python.computer_vision.compute_affinity import compute_affinity
from python.computer_vision.rect_overlap import rect_overlap
import scipy.io

t0 = time()
################## Parameter Settings ########################
# Size of patches around interesting points
sz_patch = (25-1)/2

# Number of clusters for visual words
n_clusters = 20

# Minimum distance in Harris Corner Detector Non-maximum Suppression
mindis_harris = 5

# K-Means running times
times_kmeans = 1

# Directories of dataset
base_path = "/Users/changyale/dataset/computer_vision/"
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
    feat_harris.append(feature.peak_local_max(tmp,min_distance=mindis_harris))

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

# Apply PCA to reduce dimensionality of feat_desc
pca = PCA(n_components=30)
#feat_pca = pca.fit_transform(feat_desc/255)
#print "PCA Variance Ratio:",sum(pca.explained_variance_ratio_)

# Step(c): Cluster the patches images into clusters using K-means. This step is
# meant to significantly reduce the number of possible visual words. The
# clusters that you find here constitute a "visual vocabulary".
n_clusters_range = range(n_clusters,n_clusters+1)
score = []

for n_clusters in n_clusters_range:
    
    """
    # Affinity matrix, diagonal elements should be set to be 0
    sigma,tmp = compute_affinity(feat_pca,flag_sigma='global')
    mtr_aff = tmp*(np.ones(tmp.shape)-np.eye(tmp.shape[0]))
    
    # Degree matrix
    tmp = np.diag(1./np.sum(mtr_aff,1))

    # Laplacian matrix
    mtr_lap = np.eye(tmp.shape[0])-tmp.dot(mtr_aff)
    
    eig_val,eig_vec = np.linalg.eig(mtr_lap)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    """
    clf_list = []
    inertia_list = []
    for i in range(times_kmeans):
        clf = KMeans(n_clusters=n_clusters,init='random')
        clf.fit(feat_desc/255)
        clf_list.append(clf)
        inertia_list.append(clf.inertia_)
        print i
    clf = clf_list[inertia_list.index(min(inertia_list))]
    centers = clf.cluster_centers_
    labels = clf.labels_
    #score.append(clf.inertia_)
    #print n_clusters
#plt.plot(n_clusters_range,score)
#plt.show()

# Step(d): Having found the vocabulary, go back to each training example and
# assign their local patches to visual words in the vocabulary. An image patch
# is assigned to the visual word which is closest using Euclidean distance(SSD)


# Step(e): For each visual word occurence in the training examples, record the
# possible displacement vectors between it and the object center. Assume the
# object center is the center of the cropped training image.
displacement = []
for i in range(n_clusters):
    displacement.append([])

index = 0
for i in range(len(feat_harris)):
    if feat_harris[i].shape[0]>0:
        for j in range(feat_harris[i].shape[0]):
            displacement[labels[index]].append(feat_harris[i][j,:]-\
                    np.array(img[i].shape)/2)
            index += 1

## 2. Testing; Given a novel test image, detect instances of the object
# Step(a): Run the corner detector to find interesting points
feat_harris_test = []
for i in range(len(img_test)):
    # Harris corner detector
    tmp = feature.corner_harris(img_test[i],method='k',k=0.05,eps=1e-6,\
            sigma=1.0)
    # Only keep the local maximum using nonmax suppression
    feat_harris_test.append(feature.peak_local_max(tmp,min_distance=mindis_harris))

# Step(b): At each interesting point, use a fixed image patch of the same size
# as the ones used during training to create a raw pixel descriptor
feat_desc_test = []
for i in range(len(feat_harris_test)):
    if feat_harris_test[i].shape[0]>0:
        for j in range(feat_harris_test[i].shape[0]):
            tmp = get_subwindow(img_test[i],list(feat_harris_test[i][j,:]),\
                    sz_patch)
            tmp = tmp.reshape(1,tmp.shape[0]*tmp.shape[1])
            feat_desc_test.append(tmp[0,:])
feat_desc_test = np.array(feat_desc_test)

# Step(c): Assign to each patch a visual word
labels_pred = clf.predict(feat_desc_test/255)

# Step(d): Let each visual word occurence vote for the position of the object
# using the stored displacement vectors.
"""
# votes for each visual word
votes = []
for i in range(n_clusters):
    idx_row = []
    idx_col = []
    for j in range(len(displacement[i])):
        idx_row.append(displacement[i][j][0])
        idx_col.append(displacement[i][j][1])
    idx_row = np.array(idx_row)
    bins_row = np.max(idx_row)-np.min(idx_row)
    idx_col = np.array(idx_col)
    bins_col = np.max(idx_col)-np.min(idx_col)
    hist,edge_row,edge_col = np.histogram2d(idx_row,idx_col,bins=[10,10])
    idx_peak = np.unravel_index(hist.argmax(),hist.shape)
    row_peak = np.floor((edge_row[idx_peak[0]]+edge_row[idx_peak[0]+1])/2)
    col_peak = np.floor((edge_col[idx_peak[1]]+edge_col[idx_peak[1]+1])/2)
    votes.append([row_peak,col_peak])
votes = np.array(votes)
"""

# Step(e): After all votes are cast, analyze the votes in the accumulatory
# array, threshold and predict where the object occurs. To predict the fixed
# size bounding box placement, assume that the object center is the bounding
# box center. Note that the object of interest may occur multiple times in the
# test image.
index = 0
# Predicted Location in the image
loc_pred = []
for i in range(len(feat_harris_test)):
    tmp = []
    if feat_harris_test[i].shape[0]>0:
        for j in range(feat_harris_test[i].shape[0]):
            tmp_1 = np.repeat(np.array([feat_harris_test[i][j]]),\
                    len(displacement[labels_pred[index]]),axis=0)
            tmp_1 = tmp_1-np.array(displacement[labels_pred[index]])
            tmp = tmp+list(tmp_1)
            index += 1
    idx_row = []
    idx_col = []
    for k in range(len(tmp)):
        idx_row.append(tmp[k][0])
        idx_col.append(tmp[k][1])
    idx_row = np.array(idx_row)
    bins_row = np.max(idx_row)-np.min(idx_row)
    idx_col = np.array(idx_col)
    bins_col = np.max(idx_col)-np.min(idx_col)
    hist,edge_row,edge_col = np.histogram2d(idx_row,idx_col,\
            bins=[40,100])
    idx_peak = np.unravel_index(hist.argmax(),hist.shape)
    loc_peak = np.floor(np.array([edge_row[idx_peak[0]],\
            edge_col[idx_peak[1]]]))
    loc_pred.append(np.array([loc_peak[1]-50,loc_peak[0]-20]))
#for i in range(len(loc_pred)):
#    print i+1,loc_pred[i]

# Step(f): Compute the accuracy of the predictions, based on the overlap
# between the predicted and true bounding boxes. Specifically, a predicted
# detection is counted as correct if the area of the intersection of the boxes,
# normalized by the area of their union exceeds 0.5
# Load ground-truth
mat = scipy.io.loadmat(base_path+ground_truth+\
        "CarsGroundTruthBoundingBoxes.mat")
x = mat['groundtruth']
truth = []
cnt = 0
for i in range(x[0].shape[0]):
    truth.append(x[0][i][0])
    rect_pred = [list(loc_pred[i]),100,40]
    if truth[i].shape[0] == 1:
        rect_true = [list(truth[i][0]),100,40]
        area_overlap = rect_overlap(rect_pred,rect_true)
        area_union = 100*40*2-area_overlap
        ratio = area_overlap*1.0/area_union
        if ratio > 0.5:
            cnt += 1
            #print i+1
    else:
        for j in range(truth[i].shape[0]):
            rect_true = [list(truth[i][j]),100,40]
            area_overlap = rect_overlap(rect_pred,rect_true)
            area_union = 100*40*2-area_overlap
            ratio = area_overlap*1.0/area_union
            if ratio > 0.5:
                cnt += 1
                #print i+1

print "Accuracy:",cnt*1./100
print "RunningTime:",time()-t0
