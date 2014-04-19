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

# Parameter Settings



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


