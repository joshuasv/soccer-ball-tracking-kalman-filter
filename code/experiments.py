#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:59:22 2022

@author: Joshua S.
@email: jsoutelo@alumnos.uvigo.es
"""

import numpy as np
import matplotlib.pyplot as plt


def get_center_yolo_coords(pos):
    return [int(pos[0] + (pos[2]-pos[0])//2), int(pos[1] + (pos[3]-pos[1])//2)]


def get_measurement_from_origin(pos):
    return [np.sqrt((0-pos[0])**2 + (0-pos[1])**2)]


def plot_distances(distances):
    plt.plot(distances)
    plt.show()

if __name__ == '__main__':
    
    gt_ball_centers = np.load('yolov5x-ball-pos-xmin-ymin-xmax-ymax.npy')
    gt_ball_centers = np.apply_along_axis(get_center_yolo_coords, 1, gt_ball_centers)
    
    pred_ball_centers = np.load('kalman-ball-centers-x-y.npy')
    
    gt_distance = np.apply_along_axis(get_measurement_from_origin, 1, gt_ball_centers).flatten()
    gt_indices_zero = np.argwhere(gt_distance==0)
    print(gt_indices_zero)
    gt_distance[gt_indices_zero] = np.nan

#    plot_distances(gt_distance)
    
    pred_distance = np.apply_along_axis(get_measurement_from_origin, 1, pred_ball_centers).flatten()
#    pred_distance[gt_indices_zero] = np.nan
    
#    plot_distances(pred_distance)
    
    fig, axx = plt.subplots(2, 1, figsize=(10,10))
    axx[0].set_title('a)')
    axx[0].set_xlabel('Frame')
    axx[0].set_ylabel('Dist (pixel)')
    axx[0].plot(gt_distance)
    axx[1].set_title('b)')
    axx[1].plot(pred_distance)
    axx[1].set_xlabel('Frame')
    axx[1].set_ylabel('Dist (pixel)')
    plt.savefig('ball-pos-gt-pred.png')
    plt.show()
    