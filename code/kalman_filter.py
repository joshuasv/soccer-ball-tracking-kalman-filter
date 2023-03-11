#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:46:25 2022

@author: Joshua S.
@email: jsoutelo@alumnos.uvigo.es
"""
import cv2
from ground_detection import ground_detection
from needless_objects_elimination import remove_objects
from player_ball_separation import ball_player_separation, display_bboxs
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def predict(x, P, A, Q=0, B=0, u=0):
    # Predicted state mean (prior mean)
    x_hat = np.dot(A, x) + np.dot(B, u)
    # Predicted state covariance
    p_hat = np.dot(np.dot(A, P), A.T) + Q
    
    return x_hat, p_hat


def correction(z, H, x, P, R=0):
    # Residual
    y = z - np.dot(H, x)
    # Project system uncertainty into measurement space
#    print(H.shape, P.shape, R.shape)
    S = np.dot(np.dot(H, P), H.T) + R
    # Map system uncertainty into Kalman gain
    K = np.dot(np.dot(P, H.T), np.linalg.pinv(S))
    # Correct x with residual, scaled by Kalman gain
    x_corr = x + np.dot(K, y)
    # Correct covariance
    KH = np.dot(K, H)
    I = np.eye(KH.shape[0])
    P_corr = np.dot(I - KH, P)
    
    return x_corr, P_corr
    

def get_center_bbox(bbox):
    x, y, w, h = bbox
    cent_x = x + (w // 2)
    cent_y = y + (h // 2)
    
    return cent_x, cent_y

def get_nearest_bbox(bboxs, prev_ball_center):
    best_candidate = None
    smallest_distance = np.inf
    prev_cent_x, prev_cent_y = prev_ball_center
    for bbox in bboxs:
        # Get bbox center
        cent_x, cent_y = get_center_bbox(bbox)
        # Compute Euclidean distance
        dist = np.sqrt((cent_x - prev_cent_x)**2 + (cent_y - prev_cent_y)**2)
        if dist < smallest_distance:
            smallest_distance = dist
            best_candidate = bbox
        
    return  best_candidate, smallest_distance


def get_candidate(pos, search_area, ball_bboxs, player_bboxs, frame_w, frame_h):
    best_candidate = None
    smallest_distance = np.inf
    pos_x, pos_y = pos
    limit_x_left = pos_x - search_area if (pos_x - search_area) > 0 else 0
    limit_x_right = pos_x + search_area if (pos_x + search_area) < frame_w else frame_w - 1
    limit_y_top = pos_y - search_area if (pos_y - search_area) > 0 else 0
    limit_y_bot = pos_y + search_area if (pos_y + search_area) < frame_h else frame_h - 1
    for bbox in ball_bboxs:
        x, y, w, h = bbox
        if x >= limit_x_left and x <= limit_x_right and y >= limit_y_top and y <= limit_y_bot:
            return 'ball', get_center_bbox(bbox), None
        
        
    for bbox in player_bboxs:
        x, y, w, h = bbox
        if x >= limit_x_left and x <= limit_x_right and y >= limit_y_top and y <= limit_y_bot:
            # Diagonal length player area
            dpla = np.sqrt(w*w + h*h)
            return 'player', get_center_bbox(bbox), dpla
        
    return None, (None, None), None
        


def regular_kalman_filter(show=False, save=False):
    prev_x = np.array([467, 1, 218, 1])
    P = np.zeros((4,4))
    # Transition matrix
    A = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]])
    # Measurement matrix
    H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]])
    Q = np.eye(4)
    ball_centers = []
    total_time = 0
    ball_centers.append([prev_x[0], prev_x[2]])
    data_path = 'data/frames-video00.mp4/'
    for fname in sorted(os.listdir(data_path), key=lambda x: int(x.split('.')[0]))[1:]:
    
        frame_nb = fname.split('.')[0]
        frame = cv2.imread(os.path.join(data_path, fname))
        frame = cv2.resize(frame, (592, 320))
        
        start = time.time()
        
        # Get ball and player bounding boxes
        ccl = ground_detection(frame)
        clean_ccl = remove_objects(ccl, frame)
        player_bboxs, ball_bboxs = ball_player_separation(clean_ccl)
        
        # Predict where the ball might be
        x_hat, P_hat = predict(x=prev_x, P=P, A=A, Q=Q, B=0, u=0)
        
        nearest_bbox, distance = get_nearest_bbox(ball_bboxs, np.dot(H, x_hat))
        # Measurement
        if nearest_bbox is not None or distance < 10: 
            z = get_center_bbox(nearest_bbox)
            x_corr, P_corr = correction(z=z, H=H, x=x_hat, P=P_hat, R=0)
            
            new_ball_pos = np.dot(H, x_corr)
            
            P = P_corr
            prev_x = x_corr

        else: # Rely on prediction, track the closest player
            new_ball_pos = np.dot(H, x_hat)
            P = P_hat
            prev_x = x_hat
            
            nearest_bbox, distance = get_nearest_bbox(player_bboxs, np.dot(H, x_hat))
#            print(nearest_bbox, distance)
        
        end = time.time()
        elapsed = end - start
        total_time += elapsed
        
        ball_centers.append([new_ball_pos[0], new_ball_pos[1]])
        if show:
            plt.figure(figsize=(10,10))
            frame = display_bboxs(frame, player_bboxs, ball_bboxs, size=1)
            plt.imshow(frame[...,::-1])
            plt.scatter(x=new_ball_pos[0], y=new_ball_pos[1], marker='x', s=24, c='black') 
            plt.show()
        
    ball_centers = np.array(ball_centers)
    
    if save:
        np.save('kalman-ball-centers-x-y.npy', ball_centers)
    
    n_frames = len(os.listdir(data_path))
    print(total_time, 's ;;', n_frames/total_time, 'frames/s')




def dynamic_kalman_filter():
    filter_output = []
    Q_ = lambda q: np.eye(4) * q
    R_ = lambda r: np.eye(4) * r
    # Dont know how correlated x, y positions and velocities are, covariances set to zero
    P = np.zeros((4,4))
    # Transition matrix
    A = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]])
    # Measurement matrix
    H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]])
    # Define the searh areas
    # dpla stands for diagonal lenght of player area
    search_areas = {'sm': 20, 'lg': 30, 'player': lambda dpla: int((dpla/2)+20)}
    # Manually set a correction (coords of the ball center in the first frame)
    x = np.array([467, 1, 218, 1])
    frame_w, frame_h = 592, 320
    # We don't know where the ball is
#    np.random.seed(42)
#    x = np.array([np.random.randint(frame_w), 1, np.random.randint(frame_h), 1])
    filter_output.append([x[0], x[2]])

    # We set it to small because we know that the first correspond to a ball...
    # We are in the measurement mode
    sa = 'sm'
    target = 'ball'
    Q = Q_(1)
    R = R_(0)
    dpla = 0
    data_path = 'data/frames-video00.mp4/'
    for fname in sorted(os.listdir(data_path), key=lambda x: int(x.split('.')[0]))[1:]:
        # Make a prediction
        x_hat, P_hat = predict(x=x, P=P, A=A, Q=Q, B=0, u=0)
        
        # Extract player and ball bounding boxes
        frame_nb = fname.split('.')[0]
        frame = cv2.imread(os.path.join(data_path, fname))
        frame = cv2.resize(frame, (592, 320))
        # Get ball and player bounding boxes
        ccl = ground_detection(frame)
        clean_ccl = remove_objects(ccl, frame)
        player_bboxs, ball_bboxs = ball_player_separation(clean_ccl)
        
        # Search for the nearest ball candidate
        if sa == 'player':
            search_area = search_areas[sa](dpla)
        else:
            search_area = search_areas[sa]
        cand_type, (cand_x, cand_y), dpla = get_candidate(pos=(x_hat[0], x_hat[2]),
                      search_area=search_area, 
                      ball_bboxs=ball_bboxs, player_bboxs=player_bboxs, 
                      frame_w=frame_w, frame_h=frame_h)
        
        # Adjust paramters by conditions
        if cand_type == 'ball': # We are on measurement mode
            Q = Q_(1)
            R = R_(0)
            sa = 'sm'
            target = 'ball'
        elif cand_type == 'player': # We are on player occlusion mode
            Q = Q_(1)
            R = R_(0)
            sa = 'player'
            target = 'player'
        else: # We are on prediction mode
            Q = Q_(0)
            R = R_(9999)
            sa = 'lg'
            target = 'ball'
            
        # Measurement
        if cand_x is not None and cand_y is not None:
            z = np.array([cand_x, cand_y])
        else: # We can not trust the measurement, use the prediction
            z = np.array([0, 0])
            
        # Correction
        x, P = correction(z=z, H=H, x=x_hat, P=P_hat, R=R)
        
        plt.figure(figsize=(10,10))
#            plt.title(f'{distance=}')
#            frame = display_bboxs(frame, player_bboxs, ball_bboxs)
        plt.imshow(frame[...,::-1])
        plt.scatter(x=x[0], y=x[2], marker='x', s=14, c='r') 
        plt.show()
#        break

        
    

if __name__ == '__main__':
    regular_kalman_filter(show=True)

    """dynamic_kalman_filter() gives the following error:
        ValueError: operands could not be broadcast together with shapes (2,2) (4,4) 
        
    The problem is discussed in more detail in the paper."""
#    dynamic_kalman_filter()
    
