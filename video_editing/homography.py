import os
import cv2

import numpy as np
import random
from math import floor
from matplotlib import pyplot as plt
from numpy.linalg import svd, inv

def auto_homography(Ia,Ib, homography_func=None,normalization_func=None, debug=False):
    '''
    Computes a homography that maps points from Ia to Ib

    Input: Ia and Ib are images
    Output: H is the homography

    '''
    if Ia.dtype == 'float32' and Ib.dtype == 'float32':
        Ia = (Ia*255).astype(np.uint8)
        Ib = (Ib*255).astype(np.uint8)
    
    Ia_gray = cv2.cvtColor(Ia,cv2.COLOR_BGR2GRAY)
    Ib_gray = cv2.cvtColor(Ib,cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp_a, des_a = sift.detectAndCompute(Ia_gray,None)
    kp_b, des_b = sift.detectAndCompute(Ib_gray,None)    
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a,des_b, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
   
    numMatches = int(len(good))

    matches = good

    # Xa and Xb are 3xN matrices that contain homogeneous coordinates for the N
    # matching points for each image
    Xa = np.ones((3,numMatches))
    Xb = np.ones((3,numMatches))
    
    for idx, match_i in enumerate(matches):
        Xa[:,idx][0:2] = kp_a[match_i.queryIdx].pt
        Xb[:,idx][0:2] = kp_b[match_i.trainIdx].pt

    ## RANSAC
    niter = 1000
    best_score = 0

    H = None
    for t in range(niter):
        # estimate homography
        subset = np.random.choice(numMatches, 4, replace=False)
        pts1 = Xa[:,subset]
        pts2 = Xb[:,subset]
        
        H_t = homography_func(pts1, pts2, normalization_func) # edit helper code below (computeHomography)

        # score homography
        Xb_ = np.dot(H_t, Xa) # project points from first image to second using H
        du = Xb_[0,:]/Xb_[2,:] - Xb[0,:]/Xb[2,:]
        dv = Xb_[1,:]/Xb_[2,:] - Xb[1,:]/Xb[2,:]

        ok_t = np.sqrt(du**2 + dv**2) < 1  # you may need to play with this threshold
        score_t = sum(ok_t)

        if score_t > best_score:
            best_score = score_t
            H = H_t
            in_idx = ok_t
    
    if(debug):
        print('best score: {:02f}'.format(best_score))

    # Optionally, you may want to re-estimate H based on inliers

    return H

def normalizeCoordinates(pts):
    
    trans = np.zeros((3,3),dtype=float)
    scale = np.zeros((3,3),dtype=float)
    
    std_dev_u = np.ndarray.std(pts[:,0])
    std_dev_v = np.ndarray.std(pts[:,1])
    mean_u = np.mean(pts[:,0])
    mean_v = np.mean(pts[:,1])
    
    trans[0,0] = 1/std_dev_u
    trans[1,1] = 1/std_dev_v
    trans[2,2] = 1
    scale[0,0] = 1 
    scale[0,2] = -mean_u
    scale[1,1] = 1
    scale[1,2] = -mean_v
    scale[2,2] = 1
    
    pts_norm = np.dot((trans * scale), pts)
    
    return pts_norm
    
def computeHomography(pts1, pts2, normalization_func=None):
    '''
    Compute homography that maps from pts1 to pts2 using SVD
     
    Input: pts1 and pts2 are 3xN matrices for N points in homogeneous
    coordinates. 
    
    Output: H is a 3x3 matrix, such that pts2~=H*pts1
    '''
    y, x = pts1.shape
    A = np.zeros((2*x, 9))
    
    for i in range(x):
        k = 2*i 
        A[k][0] = -pts1[0][i]
        A[k][1] = -pts1[1][i]
        A[k][2] = -1
        A[k][6] = pts1[0][i] * pts2[0][i]
        A[k][7] = pts1[1][i] * pts2[0][i]
        A[k][8] = pts2[0][i]
        
        A[k+1][3] = -pts1[0][i]
        A[k+1][4] = -pts1[1][i]
        A[k+1][5] = -1
        A[k+1][6] = pts1[0][i] * pts2[1][i]
        A[k+1][7] = pts1[1][i] * pts2[1][i]
        A[k+1][8] = pts2[1][i]
        
    u, s, vh = np.linalg.svd(A)
    rows, cols = vh.shape
    
    H = vh[:][cols-1].reshape(3,3)
    
    return H

    
