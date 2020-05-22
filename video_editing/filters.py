import cv2
import numpy as np

def resize_image(im, new_size):
    
    im_r, im_g, im_b = cv2.split(im)
    
    im_r_resized = cv2.resize(im_r, new_size)
    im_g_resized = cv2.resize(im_g, new_size)
    im_b_resized = cv2.resize(im_b, new_size)
    
    im_resized = cv2.merge((im_r_resized, im_g_resized, im_b_resized))
    
    return im_resized

def apply_gaussian_filter(im, size, sigma):
    low_pass_filtered = cv2.GaussianBlur(im, (size, size), sigma)

    return low_pass_filtered

def laplacian_filter(im, size, sigma):
    
    low_pass_filtered = apply_gaussian_filter(im, size, sigma)
    high_pass_filtered = np.subtract(im, low_pass_filtered)
    
    return high_pass_filtered

def apply_laplacian_filter(im, size, sigma):
    laplacian_filtered = np.zeros(im.shape, im.dtype)
    
    if(len(im.shape) > 2):
        im_r, im_g, im_b = cv2.split(im)
        filtered = np.zeros((3,im.shape[0],im.shape[1]), im.dtype)
        
        filtered[0] = laplacian_filter(im_r, size, sigma)
        filtered[1] = laplacian_filter(im_g, size, sigma)
        filtered[2] = laplacian_filter(im_b, size, sigma)
        laplacian_filtered = cv2.merge((filtered[0],filtered[1],filtered[2]))
    else:
        laplacian_filtered = laplacian_filter(im, size, sigma)
    
    
    return laplacian_filtered


