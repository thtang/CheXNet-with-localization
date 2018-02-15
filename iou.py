import numpy as np
import pandas as pd

import collections
import os

import scipy.ndimage as ndimage
from scipy.ndimage import binary_dilation, filters

BBOX_LIST_FNAME = 'BBox_List_2017.csv'
VALID_FNAME = 'valid.txt'
CROP_DEL, RESCALE_FACTOR = 16., 4.

# class-id mapping
class_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

def IOU(xywh1, xywh2):
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2

    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    intersection = dx * dy if (dx >=0 and dy >= 0) else 0.
    
    union = w1 * h1 + w2 * h2 - intersection
    return (intersection / union)

def preprocess_bbox_df(mismatch_id=-1):
    # filename-id mapping
    with open('valid.txt', 'r') as f:
        valid_list = f.readlines()
        valid_list = [s.strip('\n') for s in valid_list]

    df = pd.read_csv(BBOX_LIST_FNAME)
    
    # map file/class name to id
    df['Image Index'] = df['Image Index'].apply(lambda x: valid_list.index(x) if x in valid_list else mismatch_id)
    #df['Finding Label'] = df['Finding Label'].apply(lambda x: class_list.index(x))
    
    return df

def validate_score(predicted_xywh, bbox_df, img_id, class_name):
    match_row = bbox_df[(bbox_df['Image Index'] == img_id) & (bbox_df['Finding Label'] == class_name)]
    try:
        assert(len(match_row)<=1)
    except:
        print('error with query:', match_row)
    
    # IOU = 0.0 for mismatch cases
    if match_row.empty:
        return 0.
    
    ground_truth_xywh = tuple(match_row.iloc[0,2:6])
    
    return IOU(predicted_xywh, ground_truth_xywh)

def validate_total_score(default_box, valid_dir='valid_heatmap'):
    npy_list = os.listdir(valid_dir)
    with open('valid.txt', 'r') as f:
        fname_list = f.readlines()
        fname_list = [s.strip('\n') for s in fname_list]

    prediction_dict = {}
    for i in range(440):
        prediction_dict[i] = []

    for npy_name in npy_list:
        if not npy_name.endswith('.npy'):
            continue
            
        data = np.load(os.path.join(valid_dir, npy_name))
        img_id = int(npy_name.split('.')[0].split('_')[1])
        k = int(npy_name.split('.')[0].split('_')[2])
            
        # predict default_box
        prediction_sent = (class_list[k], default_box[k][0], default_box[k][1], default_box[k][2], default_box[k][3])
        prediction_dict[img_id].append(prediction_sent)

        if np.isnan(data).any():
            continue
        
        img_width, img_height = 224, 224
        w_k, h_k = (default_box[k][2:] * (1 / RESCALE_FACTOR)).astype(np.int)    
    
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
        
        for pt in xy:
            if data[int(pt[0]), int(pt[1])] > np.max(data)*.9:
                upper = int(max(pt[0]-(h_k/2), 0.))
                left = int(max(pt[1]-(w_k/2), 0.))
                
                right = int(min(left+w_k, img_width))
                lower = int(min(upper+h_k, img_height))
                
                if lower == img_height and not k in [1]:
                    # avoid bbox touching bottom
                    continue
                elif k in [5]:
                    # avoid predicting low acc classes
                    continue
                else:
                    prediction_sent = (class_list[k], (left+CROP_DEL)*RESCALE_FACTOR, (upper+CROP_DEL)*RESCALE_FACTOR,
                                       (right-left)*RESCALE_FACTOR, (lower-upper)*RESCALE_FACTOR)
                
                prediction_dict[img_id].append(prediction_sent)
    
    # calculate IOU score
    bbox_df = preprocess_bbox_df()
    
    iou_sum = 0.
    iou25_count, iou50_count = 0, 0
    box_count = 0

    for img_id in range(len(prediction_dict)):
        for pred in prediction_dict[img_id][:10]:
            iou = validate_score(pred[1:], bbox_df, img_id, pred[0])
                                 
            iou_sum += iou
            iou25_count += 1 if iou > .25 else 0
            iou50_count += 1 if iou > .50 else 0
            box_count = box_count + 1
        
    iou_avg = iou_sum / box_count
    iou25_avg = iou25_count / box_count
    iou50_avg = iou50_count / box_count
    iou_score = (iou25_avg + iou50_avg) / 2

    #print('total box_count =', box_count)
    #print('average IOU =', iou_avg)
    #print('average score at T(0.25) =', iou25_avg)
    #print('average score at T(0.50) =', iou50_avg)
    
    return iou_score

if __name__ == '__main__':
    print(IOU((3., 3., 2., 2.), (1., 1., 3., 2.5)))
