# Find the size of bounding boxes
# Usage: python3 find_bbox_size.py <heatmap_dir>

# <heatmap_dir> contains .npy files named 'heatmap_<id>_<class>.npy',
# descripting the heatmap of positive cases predicted by our model
#
# <id> is the order of each validation image (in valid.txt)
# <class> is the disease class in alphabetical order (0-7)

# Available .npy files (heatmaps) can be download at:
# https://drive.google.com/file/d/1A-FUIfnBkBqvxtncNRYtOLgVgZUOZIl_/view

import numpy as np
import pandas as pd
import sys, os

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation

crop_del, rescale_factor = 16, 4

class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
intensity_th = [0.85, 0.50, 0.70, 0.70, 0.85, 0.80, 0.70, 0.70]
# heuristic threshold: all started from 1.00, tuned by 0.05 each time to find better threshold value

x_avg = np.zeros(8)
y_avg = np.zeros(8)
w_avg = np.zeros(8)
h_avg = np.zeros(8)
class_count = np.zeros(8)

npy_list = os.listdir(sys.argv[1])


for npy_name in npy_list:
    if not npy_name.endswith('.npy'):
        continue

    data = np.load(os.path.join(sys.argv[1],npy_name))

    img_id = int(npy_name.split('.')[0].split('_')[1])
    img_fname = fname_list[img_id]
    k = int(npy_name.split('.')[0].split('_')[2])

    if np.isnan(data).any():
        continue
    
    # Find local maxima
    neighborhood_size = 100
    threshold = .1
    
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    for _ in range(5):
        maxima = binary_dilation(maxima)
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
    
    thresholded_data = (data > np.max(data)*intensity_th[k]).astype(np.int)
    
    for pt in xy:
        if data[int(pt[0]), int(pt[1])] > np.max(data)*.9:
            # find box boundaries
            center_coord = [int(pt[0]), int(pt[1])]
            left, right, upper, lower = center_coord[0], center_coord[0], center_coord[1], center_coord[1]
            
            while left > 0 and thresholded_data[max(0,left), center_coord[1]] == 1:
                left -= 1
            while right < 224 and thresholded_data[min(224,right), center_coord[1]] == 1:
                right += 1
            while upper > 0 and thresholded_data[center_coord[0], max(0,upper)] == 1:
                upper -= 1
            while lower < 224 and thresholded_data[center_coord[0], min(224,lower)] == 1:
                lower += 1
            
            x_avg[k] += left
            y_avg[k] += upper
            w_avg[k] += right-left
            h_avg[k] += lower-upper
            class_count[k] += 1
            
# output stats
x_avg = (x_avg/class_count + 16.) * 4.
y_avg = (y_avg/class_count + 16.) * 4.
w_avg = w_avg/class_count * 4.
h_avg = h_avg/class_count * 4.

for k in range(8):
    print('[%.1f, %.1f, %.1f, %.1f]'%(x_avg[k], y_avg[k], w_avg[k], h_avg[k]))
