import os
import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from utils import display_images

def normalize_img(img):
    
    min_val = np.max(img)
    max_val = np.max(img)
    c = max_val / (1 + max_val)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (img[y][x] > 1.0):
                img[y][x] = 1.0
            if (img[y][x] < 0.0):
                img[y][x] = 0.0
                
    return img


def gradient_descent(src_img, mask, trgt_img):
    """
    The implementation for gradient domain processing is not complicated, but it is easy to make a mistake, so let's start with a toy example. Reconstruct this image from its gradient values, plus one pixel intensity. Denote the intensity of the source image at (x, y) as s(x,y) and the value to solve for as v(x,y). For each pixel, then, we have two objectives:
    1. minimize (v(x+1,y)-v(x,y) - (s(x+1,y)-s(x,y)))^2
    2. minimize (v(x,y+1)-v(x,y) - (s(x,y+1)-s(x,y)))^2
    3. minimize (v(x+1,y)-t(x,y) - (s(x+1,y)-s(x,y)))^2
    4. minimize (v(x,y+1)-t(x,y) - (s(x,y+1)-s(x,y)) )^2
    
    :param im: numpy.ndarray
    """
    #print("Src Max:", np.max(src_img), " Trgt Max: ", np.max(trgt_img))
    
    im_h, im_w = trgt_img.shape
    
    matrix_size = 2*((im_h)*(im_w-1) + (im_h-1)*(im_w))
    
    #print("Matrix size: ", matrix_size)
        
    A = lil_matrix((matrix_size, im_h * im_w)) 
    b = np.zeros((matrix_size))
    
    im2var = np.arange(im_h * im_w).reshape(im_h, im_w)
    
    e = -1
    
    for y in range(im_h): 
        for x in range(im_w):
            
            if (x + 1 < im_w):
                e += 1
                A[e, im2var[y][x+1]] = 1
                A[e, im2var[y][x]] = -1
                b[e] = src_img[y][x+1] - src_img[y][x]
             
                if(mask[y][x] == 0):
                    e += 1
                    A[e, im2var[y][x+1]] = -1
                    b[e] = -src_img[y][x+1] + src_img[y][x] - trgt_img[y][x]
             
            if (y + 1 < im_h):
                e += 1    
                A[e, im2var[y+1][x]] = 1
                A[e, im2var[y][x]] = - 1
                b[e] = src_img[y+1][x] - src_img[y][x] 
                
                if(mask[y][x] == 0):
                    e += 1
                    A[e, im2var[y+1][x]] = -1
                    b[e] = -src_img[y+1][x] + src_img[y][x] - trgt_img[y][x]
        
                
    #print("No of equations: ", e)
    
    #assert matrix_size > (e+1)
    
    var = lsqr(A.tocsr(), b)
    result = var[0].reshape(im_h, im_w)
   
    result = normalize_img(result)
    
    return result

def find_blending_region_new(mask, src_img_arr, trgt_img_arr):

    H,W = src_img_arr[0].shape

    # start_y, end_y, start_x, end_x, new_mask, new_src, new_trgt = 
    return 0, H, 0, W, mask, src_img_arr, trgt_img_arr

def find_blending_region(mask, src_img_arr, trgt_img_arr):
    
    im_h, im_w = mask.shape
    
    last_blank = 0
    
    start_y = -1
    end_y = -1

    # Wait till the first blank row till next set of blank rows
    
    for y in range(im_h):
        mask_sum = np.sum(mask[y][:])
        
        if (mask_sum == 0):
            if (start_y != -1):
                #print("Mask Y startpoint recognised")
                if(last_blank + 1 == y):
                    end_y = y
                    #print("Last non blank row:", y)
                    break
                        
            last_blank = y

        else:
            if (start_y == -1):
                #print("First non blank row:", y)
                start_y = y
    
    if(start_y == -1):
        start_y = 0
    if (end_y == -1):
        end_y = im_h
        
    #print("start_y: ", start_y, " end_y: ", end_y)  
    
    last_blank = 0
    start_x = -1
    end_x = -1
    
    mask_T = np.transpose(mask)
    for x in range(im_w):
        mask_sum = np.sum(mask_T[x][:])
        
        if (mask_sum == 0):
            if (start_x != -1):
                #print("Mask X startpoint recognised")
                if(last_blank + 1 == x):
                    end_x = x
                    #print("Last non blank column:", x)
                    break
                        
            last_blank = x

        else:
            if (start_x == -1):
                #print("First non blank column:", x)
                start_x = x
    
    if(start_x == -1):
        start_x = 0
    if (end_x == -1):
        end_x = im_w
        
    #print("start_x: ", start_x, " end_x: ", end_x)
    
    new_h = end_y - start_y + 1
    new_w = end_x - start_x + 1
    
    new_mask = np.zeros((new_h, new_w))
    
    new_src_r = np.zeros((new_h, new_w))
    new_src_g = np.zeros((new_h, new_w))
    new_src_b = np.zeros((new_h, new_w))
    
    new_trgt_r = np.zeros((new_h, new_w))
    new_trgt_g = np.zeros((new_h, new_w))
    new_trgt_b = np.zeros((new_h, new_w))
    
   # print("Mask size: ", new_h, " X", new_w, " Shape: ", new_mask.shape, " Size: ", new_mask.size)
    
    y = start_y
    new_y = 0
    
    while(y < end_y):
        x = start_x
        new_x = 0
        while (x < end_x):
            new_mask[new_y][new_x] = mask[y][x]
            
            new_src_r[new_y][new_x] = src_img_arr[0][y][x]
            new_src_g[new_y][new_x] = src_img_arr[1][y][x]
            new_src_b[new_y][new_x] = src_img_arr[2][y][x]
            
            new_trgt_r[new_y][new_x] = trgt_img_arr[0][y][x]
            new_trgt_g[new_y][new_x] = trgt_img_arr[1][y][x]
            new_trgt_b[new_y][new_x] = trgt_img_arr[2][y][x]
            
            x += 1
            new_x += 1
            
        y += 1
        new_y += 1
    
    new_src = [new_src_r,new_src_g, new_src_b]
    new_trgt = [new_trgt_r, new_trgt_g, new_trgt_b]
    
    return start_y, end_y, start_x, end_x, new_mask, new_src, new_trgt


def poisson_blend(object_img, object_mask, background_img, grad_descent_func=gradient_descent):
    """
    :param cropped_object: numpy.ndarray One you get from align_source
    :param object_mask: numpy.ndarray One you get from align_source
    :param background_img: numpy.ndarray 
    """
    mask3d = np.zeros([object_mask.shape[0], object_mask.shape[1], 3])
    for i in range(3):
        mask3d[:,:, i] = object_mask 
    
    src_img =  object_img
    
    src_img_r, src_img_g, src_img_b = cv2.split(src_img)
    trgt_img_r, trgt_img_g, trgt_img_b = cv2.split(background_img)

    start_y, end_y, start_x, end_x, new_mask, new_src, new_trgt = find_blending_region(object_mask,\
                                                [src_img_r,src_img_g,src_img_b],\
                                                [trgt_img_r,trgt_img_g,trgt_img_b])
    

    blended_r = grad_descent_func(new_src[0], new_mask, new_trgt[0])
    blended_g = grad_descent_func(new_src[1], new_mask, new_trgt[1])
    blended_b = grad_descent_func(new_src[2], new_mask, new_trgt[2])
    interim_result = cv2.merge((blended_r, blended_g, blended_b))

    interim_src = cv2.merge((new_src))
    interim_trgt = cv2.merge((new_trgt))
   
    display_images(4, [new_mask, interim_src, interim_trgt, interim_result], \
                   ['Mask ','Object', 'Target Region','Interim Result'])
    
    
    im_h, im_w = trgt_img_r.shape 
    
    out_img_r = np.ones((im_h, im_w))
    out_img_g = np.ones((im_h, im_w))
    out_img_b = np.ones((im_h, im_w))
    
    #print("Mask coordinates: ", start_y,"-",end_y,", ", start_x, "-", end_x)
    
    #Overwrite the masked region with the blend results
    h,w = blended_r.shape
    y = 0
    while(y < h and y+start_y < im_h):
        x = 0
        while(x < w and x+start_x < im_w):
            out_img_r[y+start_y][x+start_x] = blended_r[y][x]
            out_img_g[y+start_y][x+start_x] = blended_g[y][x]
            out_img_b[y+start_y][x+start_x] = blended_b[y][x]
            x += 1
            
        y += 1
    
    #print("Blended: Max(r): ", np.max(out_img_r), " , max(g):", np.max(out_img_g), " max(b): ", np.max(out_img_b))
    
    
    #Final result = mask * blended region + (1-mask) * background image
    out_img_r = out_img_r * object_mask + trgt_img_r * (1-object_mask)
    out_img_g = out_img_g * object_mask + trgt_img_g * (1-object_mask)
    out_img_b = out_img_b * object_mask + trgt_img_b * (1-object_mask)
    
    #print("Out: Max(r): ", np.max(out_img_r), " , max(g):", np.max(out_img_g), " max(b): ", np.max(out_img_b))
    
    out_img_rgb = cv2.merge((out_img_r, out_img_g, out_img_b)) 
    out_img_bgr = cv2.merge((out_img_b, out_img_g, out_img_r))
    
    
    return out_img_rgb, out_img_bgr


def mixed_gradient_descent(src_img, mask, trgt_img):
    """
    The implementation for gradient domain processing is not complicated, but it is easy to make a mistake, 
    so let's start with a toy example. Reconstruct this image from its gradient values, plus one pixel intensity. 
    Denote the intensity of the source image at (x, y) as s(x,y) and the value to solve for as v(x,y). For each pixel, then, we have two objectives:
    1. minimize (v(x+1,y)-v(x,y) - max(abs(s(x+1,y)-s(x,y)),abs(t(x+1,y)-t(x,y))))^2
    2. minimize (v(x,y+1)-v(x,y) - max(abs(s(x,y+1)-s(x,y)),abs(t(x,y+1)-t(x,y))))^2
    3. minimize (v(x+1,y)-t(x,y) - max(abs(s(x+1,y)-s(x,y)),abs(t(x+1,y)-t(x,y))))^2
    4. minimize (v(x,y+1)-t(x,y) - max(abs(s(x,y+1)-s(x,y)),abs(t(x,y+1)-t(x,y))))^2
    
    :param im: numpy.ndarray
    """
    im_h, im_w = trgt_img.shape
    
    matrix_size = 2*((im_h)*(im_w-1) + (im_h-1)*(im_w))
    
    #print("Matrix size: ", matrix_size)
        
    A = lil_matrix((matrix_size, im_h * im_w)) 
    b = np.zeros((matrix_size))
    
    im2var = np.arange(im_h * im_w).reshape(im_h, im_w)
    
    e = -1
    
    for y in range(im_h): 
        for x in range(im_w):
            if (x + 1 < im_w):
                e += 1
                A[e, im2var[y][x+1]] = 1
                A[e, im2var[y][x]] = -1
                
                src_gradient = src_img[y][x+1] - src_img[y][x]
                trgt_gradient = trgt_img[y][x+1]- trgt_img[y][x]
                
                if (abs(src_gradient)> abs(trgt_gradient)):
                    b[e] = src_gradient
                else:
                    b[e] = trgt_gradient
             
                if(mask[y][x] == 0):
                    e += 1
                    A[e, im2var[y][x+1]] = -1
                    if (abs(src_gradient)> abs(trgt_gradient)):
                        b[e] = src_gradient - trgt_img[y][x]
                    else:
                        b[e] = trgt_gradient - trgt_img[y][x]
             
            if (y + 1 < im_h):
                e += 1    
                A[e, im2var[y+1][x]] = 1
                A[e, im2var[y][x]] = - 1
                
                src_gradient = src_img[y+1][x] - src_img[y][x]
                trgt_gradient = trgt_img[y+1][x] - trgt_img[y][x]
                
                if(abs(src_gradient) > abs(trgt_gradient)):
                    b[e] = src_gradient
                else:
                    b[e] = trgt_gradient
                 
                if(mask[y][x] == 0):
                    e += 1
                    A[e, im2var[y+1][x]] = -1
                    
                    if(abs(src_gradient) > abs(trgt_gradient)):
                        b[e] = src_gradient - trgt_img[y][x]
                    else:
                        b[e] = trgt_gradient - trgt_img[y][x]
                
    
    var = lsqr(A.tocsr(), b)
    result = var[0].reshape(im_h, im_w)    
    
    return normalize_img(result)

def mix_blend(object_img, object_mask, background_img):
    """
    :param cropped_object: numpy.ndarray One you get from align_source
    :param object_mask: numpy.ndarray One you get from align_source
    :param background_img: numpy.ndarray 
    """

    im_blend_rgb, im_blend_bgr = poisson_blend(object_img, object_mask, background_img, mixed_gradient_descent)

    display_images(4, [ object_img, object_mask, background_img, im_blend_rgb], \
                       ['Object','Mask', 'Background','Blended'])


    return im_blend_rgb, im_blend_bgr

