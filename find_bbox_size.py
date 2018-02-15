# Find the size of bounding boxes
# Usage: python3 find_bbox_size.py <train_heatmap> <valid_heatmap>

# <{train,valid}_heatmap> contains .npy files named 'heatmap_<id>_<class>.npy'
# describing the heatmap of positive cases predicted by our model
# files in <valid_heatmap> will only be used to calculate IOU scores
#
# <id> is the order of each validation image (in valid.txt)
# <class> is the disease class in alphabetical order (0-7)

import numpy as np
import pandas as pd
import sys, os

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation

from iou import validate_total_score

CROP_DEL, RESCALE_FACTOR = 16., 4.
class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

# heuristic threshold: all started from 1.00, tuned by 0.05 each time to find better threshold value
intensity_th = [1., 1., 1., 1., 1., 1., 1., 1.]
delta_th = .05

prev_default_box = None

for target_th in range(8):
    # consider each class independent
    max_iou_score = .0
    
    for _ in range(10):
        intensity_th[target_th] -= delta_th
    
        x_sum = np.zeros(8)
        y_sum = np.zeros(8)
        w_sum = np.zeros(8)
        h_sum = np.zeros(8)
        class_count = np.zeros(8)
        
        npy_list = os.listdir(sys.argv[1])
        
        for npy_name in npy_list:
            if not npy_name.endswith('.npy'):
                continue
        
            data = np.load(os.path.join(sys.argv[1], npy_name))
        
            img_id = int(npy_name.split('.')[0].split('_')[1])
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
        
            for point in xy:
                centroid_x = int(point[0])
                centroid_y = int(point[1])
                
                if data[centroid_x, centroid_y] > np.max(data)*.9:
                    # find box boundaries
                    left, right, upper, lower = centroid_x, centroid_x, centroid_y, centroid_y
                    
                    while left > 0 and thresholded_data[max(0,left), centroid_y] == 1:
                        left -= 1
                    while right < 224 and thresholded_data[min(224,right), centroid_y] == 1:
                        right += 1
                    while upper > 0 and thresholded_data[centroid_x, max(0,upper)] == 1:
                        upper -= 1
                    while lower < 224 and thresholded_data[centroid_x, min(224,lower)] == 1:
                        lower += 1
                    
                    x_sum[k] += left
                    y_sum[k] += upper
                    w_sum[k] += right - left
                    h_sum[k] += lower - upper
                    class_count[k] += 1
                    
        # output stats
        x_avg = (x_sum / class_count + CROP_DEL) * RESCALE_FACTOR
        y_avg = (y_sum / class_count + CROP_DEL) * RESCALE_FACTOR
        w_avg = w_sum / class_count * RESCALE_FACTOR
        h_avg = h_sum / class_count * RESCALE_FACTOR
        
        default_box = []
        for k in range(8):
            #print('[%.1f, %.1f, %.1f, %.1f]'%(x_avg[k], y_avg[k], w_avg[k], h_avg[k]))
            default_box.append([x_avg[k], y_avg[k], w_avg[k], h_avg[k]])
        
        iou_score = validate_total_score(np.array(default_box), valid_dir=sys.argv[2])
        prev_default_box = default_box
        print(iou_score)
        
        if iou_score >= max_iou_score:
            max_iou_score = iou_score
        else:
            intensity_th[target_th] += delta_th
            print('class %d stops at th = %f' % (target_th, intensity_th[target_th]))
            break

print(intensity_th)
print(prev_default_box)