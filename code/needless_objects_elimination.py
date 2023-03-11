#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:41:54 2022

@author: Joshua S.
@email: jsoutelo@alumnos.uvigo.es
"""
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ground_detection import ground_detection

import skimage

    
def most_dense_area(segmented):
    """Search for the most dense areas in the image.
    
    Return a window around the positions of high density.
    """
    empty_mat = np.zeros_like(segmented)
    mean_filter = cv2.blur(segmented, (15, 15))
    
    conts, _ = cv2.findContours(mean_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cont in conts:
        x, y, w, h = cv2.boundingRect(cont)
        
        area = w * h
        if h > w and area <= 5000 and area > 100:
            extra_h = np.floor(((h * 1.55) - h) / 2).astype(int)
            extra_w = np.floor(((w * 1.55) - w) / 2).astype(int)
 
            y = y - extra_h if y - extra_h > 0 else 0
            x = x - extra_w if x - extra_w > 0 else 0
            w = w + extra_w if w + extra_w < segmented.shape[1] else segmented.shape[1]
            h = h + extra_h if h + extra_h < segmented.shape[0] else segmented.shape[0]
         
            empty_mat[y:y+h, x:x+w] = segmented[y:y+h, x:x+w]

    return empty_mat
        
    
def remove_public_score(frame_lbl, region_props, debug=False):
    """Remove the public seats and score bar.
    """
    h, w = frame_lbl.shape[:2]
    
    for region in region_props:
        bbox = region.bbox
        area = region.area
        ecc = region.eccentricity
        extent = region.extent
        
        if bbox[1] < h * .2 and area > 40000 and ecc > .95 and extent > .85:
            label = region.label
            
            if debug:
                plt.title(f'Score and public, label={label}')
                plt.imshow(np.where(frame_lbl == label, 1, 0), cmap='gray')
                plt.show()
                print(region.label, region.area, region.eccentricity, region.extent)
                
            frame_lbl[frame_lbl == label] = 0
            break
        
    return frame_lbl
    
def remove_objects(frame_lbl, frame):
    """Remove the objects that are disconnected distinct from players and ball.
    """
    frame_lbl = frame_lbl.copy()
    region_props = skimage.measure.regionprops(label_image=frame_lbl)
    
    frame_lbl = remove_public_score(frame_lbl, region_props)
    
    area_thr = 1000
    for region in region_props:
        # Size of the object is much larger than the player size
        segmented = np.where(frame_lbl == region.label, 1, 0).astype(np.uint8)
        if region.area > area_thr:

            dense_area = most_dense_area(segmented)
            if not (dense_area == 0).all():
                dense_area *= region.label
                frame_lbl[frame_lbl == region.label] = 0
                frame_lbl += dense_area
            else:
                frame_lbl[frame_lbl == region.label] = 0
    
    return frame_lbl

if __name__ == '__main__':
    n_frame = 10
    frame = cv2.imread(f'data/frames-video00.mp4/{n_frame}.jpg')
    ccl = ground_detection(frame)
    clean_ccl = remove_objects(ccl, frame)
    
    fig, axx = plt.subplots(1, 3, figsize=(10,10))
    axx[0].imshow(frame[...,::-1])
    axx[0].set_title('Original')
    axx[1].set_title('Segmentation')
    axx[1].imshow(ccl, cmap='nipy_spectral')
    axx[2].set_title('Clean segmentation')
    axx[2].imshow(clean_ccl, cmap='nipy_spectral')
    plt.show()