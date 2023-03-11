#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:54:50 2022

@author: Joshua S.
@email: jsoutelo@alumnos.uvigo.es
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage


def ground_detection(frame, debug=False):
    """Classify pixels in the frame into ground or not ground.
    
    Return the connected component labels.
    """
    frame = cv2.resize(frame, dsize=(592, 320))
    b, g, r = cv2.split(frame)
    # Ideally, 0 -> ground ;; 1 -> everything else
    ground_color_feature = np.where((g > r) & (r > b), 0, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(image=gray,
                      threshold1=255/3,
                      threshold2=255,
                      apertureSize=3,
                      L2gradient=True)
    # Section 3.2.1, equation 8
    # Inverse condition since we want to remove the background
    result = np.where((ground_color_feature == 0) & (canny == 0), 0, 1)
    result = result.astype(np.uint8)
    # Dilation so 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    morph = cv2.dilate(result, kernel, iterations=1)

    # Connected components labeling
    nl, ccl = cv2.connectedComponents(image=morph, connectivity=4)
    if debug:
        fig, axx = plt.subplots(4, 1, figsize=(20,20))
        axx[0].imshow(ground_color_feature, cmap='gray')
        axx[0].set_title('Color detection')
        axx[1].imshow(canny, cmap='gray')
        axx[1].set_title('Edge detection')
        axx[2].imshow(result, cmap='gray')
        axx[2].set_title('Ground detection result')
        axx[3].imshow(ccl, cmap='nipy_spectral')
        axx[3].set_title(f'Labeling result, {nl} lablels')
        plt.tight_layout()
        plt.show()

        
    return ccl
        
    
    
if __name__ == '__main__':
    n_frame = 20
    frame = cv2.imread(f'data/frames-video00.mp4/{n_frame}.jpg')
    ground_detection(frame, debug=True)
    
