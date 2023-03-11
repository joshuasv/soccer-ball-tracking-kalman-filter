#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:21:40 2022

@author: Joshua S.
@email: jsoutelo@alumnos.uvigo.es
"""
import cv2
import numpy as np
from ground_detection import ground_detection
from needless_objects_elimination import remove_objects
import matplotlib.pyplot as plt
from skimage.measure import regionprops

def display_bboxs(frame, player_bboxs, ball_bboxs, size=2):
    """Draw the bounding boxes with different color depending on the type of
    objet is enclosing.
    """
    frame_copy = frame.copy()
    for bbox in player_bboxs:
        x,y,w,h = bbox
        cv2.rectangle(frame_copy, (x,y), (x+w,y+h), (255, 0, 0), size)
    for bbox in ball_bboxs:
        x,y,w,h = bbox
        cv2.rectangle(frame_copy, (x,y), (x+w,y+h), (0, 0, 255), size)
        
    return frame_copy
    

def ball_player_separation(clean_ccl, debug=False):
    """Separate the connected components into players and ball candidates.
    """
    clean_ccl = clean_ccl.copy()
    frame_h, frame_w = clean_ccl.shape[:2]
    region_props = regionprops(clean_ccl)
    player_bboxs = []
    ball_bboxs = []
    for region in region_props:
        min_row, min_col, max_row, max_col = region.bbox
        x = min_col
        y = min_row
        w = max_col - x
        h = max_row - y
        # Player and ball characteristics defined in section 3.2.3
        if (w/h) < 2 and (frame_h/16) < h and h < (frame_h/4):
            if debug:
                copy = clean_ccl.copy()
                copy = np.where(copy == region.label, 1, 0)
                copy = copy.astype(np.uint8)
                copy = cv2.cvtColor(copy*255, cv2.COLOR_GRAY2RGB)
                cv2.rectangle(copy, (x,y), (x+w, y+h), (255, 0, 0), 2)
                plt.imshow(copy, cmap='gray')
                plt.show()
            player_bboxs.append((x, y, w, h))
        elif (w/h) < 2 and (w/h) > .5 and (w*h)<300:
            if debug:
                copy = clean_ccl.copy()
                copy = np.where(copy == region.label, 1, 0)
                copy = copy.astype(np.uint8)
                copy = cv2.cvtColor(copy*255, cv2.COLOR_GRAY2RGB)
                cv2.rectangle(copy, (x,y), (x+w, y+h), (255, 0, 0), 2)
                plt.imshow(copy, cmap='gray')
                plt.show()
            ball_bboxs.append((x, y, w, h))
        else:
            continue
        
    return player_bboxs, ball_bboxs
        
if __name__ == '__main__':
    n_frame = 10
    frame = cv2.imread(f'data/frames-video00.mp4/{n_frame}.jpg')
    ccl = ground_detection(frame)
    clean_ccl = remove_objects(ccl, frame)
    
    player_bboxs, ball_bboxs = ball_player_separation(clean_ccl)
    
    frame = cv2.resize(frame, (clean_ccl.shape[1], clean_ccl.shape[0])) 

    
    plt.imshow(display_bboxs(frame, player_bboxs, ball_bboxs)[...,::-1])
    plt.show()
    
    fig, axx = plt.subplots(3, 1, figsize=(10,10))
    axx[0].set_title('a)')
    axx[0].imshow(np.where(ccl != 0, 1, 0), cmap='gray')
    axx[0].axis('off')
    axx[1].set_title('b)')
    axx[1].imshow(clean_ccl, cmap='nipy_spectral')
    axx[1].axis('off')
    axx[2].set_title('c)')
    axx[2].imshow(display_bboxs(frame, player_bboxs, ball_bboxs)[...,::-1])
    axx[2].axis('off')
    plt.savefig('ground-det-clean-seg-ball-play-cand.png')
    plt.show()