#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:11:12 2019

@author: aakanksha
"""

import numpy as np
import  cv2, sys
import glob
import ntpath
sys.path.append("..")
from models.yolo_models import get_yolo_model

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3   
def bbox_iou(box1, box2):
    
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

im_width = 864
im_height = 864
obj_threshold=0.5; max_length=256;
weight_file = '/home/aakanksha/Documents/uavTracker-master/weights/trained-blackbucks-yolo.h5'
model = get_yolo_model(im_width, im_height, num_class=1)
model.load_weights(weight_file,by_name=True)
        

test_dir = '/home/aakanksha/Documents/uavTracker-master/test/'
test_images =  glob.glob( test_dir + "*.png" )
im_num=0
for imagename in test_images: 
    image = cv2.imread(imagename)
    im_num+=1
    image_h, image_w, _ = image.shape

    image = cv2.resize(image, (im_width,im_height))
    new_image = image[:,:,::-1]/255.
    new_image = np.expand_dims(new_image, 0)
    
    # get detections
    preds = model.predict(new_image)
    
    #print('yolo time: ', (stop-start)/batches)
    new_boxes = np.zeros((0,5))
    for i in range(3):
        netout=preds[i][0]
        grid_h, grid_w = netout.shape[:2]
        xpos = netout[...,0]
        ypos = netout[...,1]
        wpos = netout[...,2]
        hpos = netout[...,3]
                
        objectness = netout[...,4]
        indexes = (objectness > obj_threshold) & (wpos<max_length) & (hpos<max_length)
    
        if np.sum(indexes)==0:
            continue
    
        corner1 = np.column_stack((xpos[indexes]-wpos[indexes]/2.0, ypos[indexes]-hpos[indexes]/2.0))
        corner2 = np.column_stack((xpos[indexes]+wpos[indexes]/2.0, ypos[indexes]+hpos[indexes]/2.0))
        new_boxes = np.append(new_boxes, np.column_stack((corner1, corner2, objectness[indexes])),axis=0)
    
    # do nms 
    sorted_indices = np.argsort(-new_boxes[:,4])
    boxes=new_boxes.tolist()
    
    for i in range(len(sorted_indices)):
    
        index_i = sorted_indices[i]
        
        if new_boxes[index_i,4] == 0: 
            continue
        
        for j in range(i+1, len(sorted_indices)):
            index_j = sorted_indices[j]
            if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) >= obj_threshold:
                new_boxes[index_j,4] = 0
    
    new_boxes = new_boxes[new_boxes[:,4]>0]
    detection_list = []
    for row in new_boxes:
        stacker = (row[0],row[1],row[2],row[3], row[4])
        detection_list.append(stacker)
      
    #Display the detections    
    for detect in detection_list:
        bbox = detect[0:4]
        #if 1:
        #iwarp = (full_warp)
        #corner1 = np.expand_dims([bbox[0],bbox[1]], axis=0)
        #corner1 = np.expand_dims(corner1,axis=0)
        #corner1 = cv2.perspectiveTransform(corner1,iwarp)[0,0,:]
        minx = bbox[0]
        miny = bbox[1]
        #corner2 = np.expand_dims([bbox[2],bbox[3]], axis=0)
        #corner2 = np.expand_dims(corner2,axis=0)
        #corner2 = cv2.perspectiveTransform(corner2,iwarp)[0,0,:]
        maxx = bbox[2]
        maxy = bbox[3]
    
        cv2.rectangle(image, (int(minx)-2, int(miny)-2), (int(maxx)+2, int(maxy)+2),(0,0,0), 1)
        
      
        #write output image
    cv2.imwrite('/home/aakanksha/Documents/uavTracker-master/testOut/'+ntpath.basename(imagename),image)    