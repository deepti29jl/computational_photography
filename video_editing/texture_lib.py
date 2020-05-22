import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


def fill_patch(img_r, img_g, img_b, src_i, src_j, patch_h, patch_w, debug=False):
    
    patch = np.zeros((patch_h, patch_w,3), dtype=img_r.dtype)
     
    for i in range(patch_h):
        for j in range(patch_w):
            if(debug):
                print('i: ', str(i), ' j: ', str(j), \
                      ' src_i:', str(src_i+i), ' src_j: ', str(src_j+j),\
                      ' r: ',img_r[src_i + i][src_j + j], \
                      ' g: ', img_g[src_i + i][src_j + j],\
                      ' b: ', img_b[src_i + i][src_j + j]\
                      )
            patch[i][j][0] = img_r[src_i + i][src_j + j]
            patch[i][j][1] = img_g[src_i + i][src_j + j]
            patch[i][j][2] = img_b[src_i + i][src_j + j]
    
    return patch


def select_random_patch(img_r, img_g, img_b, img_h, img_w, patch_h, patch_w, debug=False):
    
    src_i = random.randrange(0, img_h - patch_h)
    src_j = random.randrange(0, img_w - patch_w)
   
    if(debug):
        print('select_random_patch:: '\
          ' src_i: ' , str(src_i), ' src_j: ', str(src_j), \
          ' height: ', str(patch_h), ' width: ', str(patch_w))
    
    patch = fill_patch(img_r, img_g, img_b,src_i, src_j, patch_h, patch_w, False)
    
    return patch

def generate_samples(img, patch_size, img_blurred=None):
    
    h,w,c = img.shape
    
    # Generate 10% of total possible sample
    num_samples = int(min(h-patch_size,w-patch_size) * 0.1)
    
    # Generate at least 10 samples
    if (num_samples < 10):
        num_samples = 10
   
    img_b, img_g, img_r = cv2.split(img)
    img_height, img_width, channels = img.shape
    
    samples = []
    sample_luminance=[]
    
    for i in range(num_samples):
        patch = select_random_patch(img_r, img_g, img_b, img_height, img_width, patch_size, patch_size, False)
        samples.append(patch)
        
    print('Number of samples generated:', str(len(samples)))
    
    return samples

def custom_cut(err_patch):
    """
    Compute the minimum path frm the left to right side of the patch
    
    :param err_patch: numpy.ndarray    cost of cutting through each pixel
    :return: numpy.ndarray             a 0-1 mask that indicates which pixels should be on either side of the cut
    """
    
    #print('err_patch shape:', err_patch.shape)

    # create padding on top and bottom with very large cost
    padding = np.expand_dims(np.ones(err_patch.shape[1]).T*1e10,0)
    
    #print('padding shape:', padding.shape)
    
    err_patch = np.concatenate((padding, err_patch, padding), axis=0)

    h, w = err_patch.shape
    path = np.zeros([h,w], dtype="int")
    cost = np.zeros([h,w])
    cost[:,0] = err_patch[:, 0]
    cost[0,:] = err_patch[0, :]
    cost[cost.shape[0]-1,:] = err_patch[err_patch.shape[0]-1, :]
    
    # for each column, compute the cheapest connected path to the left
    # cost of path for each row from left upper/same/lower pixel
    for x in range(1,w):
        # cost of path for each row from left upper/same/lower pixel
        tmp = np.vstack((cost[0:h-2,x-1], cost[1:h-1, x-1], cost[2:h, x-1]))
        mi = tmp.argmin(axis=0)
        path[1:h-1, x] = np.arange(1, h-1, 1).T + mi # save the next step of the path
        cost[1:h-1, x] = cost[path[1:h-1, x] - 1, x-1] + err_patch[1:h-1, x]

    path = path[1:path.shape[0]-1, :] - 1
    cost = cost[1:cost.shape[0]-1, :]
    
    # create the mask based on the best path
    mask = np.zeros(path.shape, dtype="int")
    best_path = np.zeros(path.shape[1], dtype="int")
    best_path[len(best_path)-1] = np.argmin(cost[:, cost.shape[1]-1]) + 1
    mask[0:best_path[best_path.shape[0]-1], mask.shape[1]-1] = 1
    for x in range(best_path.size-1, 0, -1):
        best_path[x-1] = path[best_path[x]-1, x]
        mask[:best_path[x-1], x-1] = 1
    mask ^= 1
    return mask


def ssd_patch_seamcut(prev_patch, new_patch, patch_size, overlap, debug=False):
    
    height = patch_size
    width = 2*(patch_size - overlap)
    
    if(debug):
        print('ssd_patch:: template height: ', str(height) , ' width: ', width)
    
    template = np.zeros((height, width, 3))
    
    for i in range(height):
        k = 0
        for j in range(patch_size - overlap):
            template[i][j][0] = prev_patch[i][j][0]
            template[i][j][1] = prev_patch[i][j][1]
            template[i][j][2] = prev_patch[i][j][2]
            k += 1
        
        j = overlap
        while j < (patch_size-overlap):
            template[i][k+j][0] = new_patch[i][j][0]
            template[i][k+j][1] = new_patch[i][j][1]
            template[i][k+j][2] = new_patch[i][j][2]
            j += 1
            
        #print('After patch 1: current i: ', str(i),' current k: ', str(k))
            
    template_T = np.transpose(template)
    
   # Horizontal Cut
    mask_1 = np.zeros(template_T.shape)
    mask_1[0] = custom_cut(template_T[0])
    mask_1[1] = custom_cut(template_T[1])
    mask_1[2] = custom_cut(template_T[2])

    mask1 = np.transpose(mask_1)
    
    # Vertical Cut
    mask_2 = np.zeros(template_T.shape)
    mask_2[0] = np.transpose(custom_cut(np.transpose(template_T[0])))
    mask_2[1] = np.transpose(custom_cut(np.transpose(template_T[1])))
    mask_2[2] = np.transpose(custom_cut(np.transpose(template_T[2])))

    mask2 = np.transpose(mask_2)
    
    #print('\nMask1 shape: ', mask1.shape)
    #print('Mask2 shape: ', mask2.shape)
    
    mask = np.bitwise_and(mask1.astype('uint8'), mask2.astype('uint8'))
    
    #print('Mask shape: ', mask.shape)
    
    '''
     Reference: Assignment Tips
     Suppose I have a template T, a mask M, and an image I: then, 
     ssd = ((M*T)**2).sum() - 2 * cv2.filter2D(I, ddepth=-1, kernel = M*T) + cv2.filter2D(I ** 2, ddepth=-1, kernel=M)
    '''
    ssd = ((mask*template)**2).sum() - 2 * cv2.filter2D(new_patch, ddepth=-1, kernel=mask*template) \
            + cv2.filter2D(new_patch ** 2, ddepth=-1, kernel=mask)
    
    
    return ssd

def select_sample_seamcut(samples, prev_patch, patch_size, overlap, tol):
    
    total_samples = len(samples)
    if (prev_patch is None):
        i = random.randrange(0, total_samples)
        return samples[i]
    
    temp_height = patch_size
    temp_width = 2*(patch_size - overlap)
    
    center_i = int(temp_height/2)
    center_j = int(temp_width/2)
    
    patch_array = []
    ssd_array = []
    cost_array = []
    center_array = []

    sel_patch = None
    sel_patch_idx = -1
    min_cost = -1
    
    for i in range(total_samples):
        
        idx = random.randrange(0, total_samples)
        
        patch = samples[idx]
        ssd = ssd_patch_seamcut(prev_patch, patch, patch_size, overlap, False)
        min_cut  = np.zeros((3, temp_height))
        
        row = 0
        col = center_j
        
        # find min path for the cut
        while (row < temp_height):
            min_cut[0][row] = ssd[row][center_j][0]
            min_cut[1][row] = ssd[row][center_j][1]
            min_cut[2][row] = ssd[row][center_j][2]
            row += 1
            
        #print (min_cut)
        cost = np.sum(min_cut)
        
        patch_array.append(patch)
        ssd_array.append(ssd)
        cost_array.append(cost)
        
        if (min_cost == -1 or cost < min_cost):
            min_cost = cost
            sel_patch = patch
            sel_patch_idx = idx
            
    #print('Selected patch with index: ', str(sel_patch_idx), ' Cost: ', '{:.2f}'.format(min_cost))
    
    return sel_patch


def quilt_cut(in_img, out_size, patch_size, overlap, tol):
    """
    Samples square patches of size patchsize from sample using seam finding in order to create an output image of size outsize.
    Feel free to add function parameters
    :param sample: numpy.ndarray
    :param out_size: int
    :param patch_size: int
    :param overlap: int
    :param tol: float
    :return: numpy.ndarray
    """
        
    if (type(out_size).__name__ != 'int' or type(patch_size).__name__ != 'int' \
        or type(overlap).__name__ != 'int' or type(tol).__name__ != 'float'):
        print('Invalid datatype for at least one of the following:\n ', \
            '\t out_size: int\n', \
            '\t patch_size: int\n', \
            '\t overlap: int\n', \
            '\t tol: float\n')
    
        return None
    if (patch_size <= overlap):
        print('patch_size value (',str(patch_size),') should be greater than overlap (', str(overlap),')!')
        return None
    
    
    h,w,c = in_img.shape

    img_b, img_g, img_r = cv2.split(in_img)

    out_img_r = np.zeros((out_size, out_size), dtype=in_img.dtype)
    out_img_g = np.zeros((out_size, out_size), dtype=in_img.dtype)
    out_img_b = np.zeros((out_size, out_size), dtype=in_img.dtype)
                      
    dst_i = 0 
    dst_j = 0
    
    samples = generate_samples(in_img/255.0, patch_size)
    prev_patch = samples[0].copy()

    max_i = out_size
    max_j = out_size
    
    print('prev_patch: ', prev_patch.shape)
    
    num_patches = 0

    while dst_i < max_i: 
    
        sample = select_sample_seamcut(samples, prev_patch, patch_size, overlap, tol) 
        num_patches += 1
        prev_patch = sample
        
        #print('quilt_simple:: sample.shape: ', sample.shape, ' sample: ', sample[:][:][0])
                
        while dst_j < max_j:
            '''
             TODO:
             
             For 'first' row, we only need to check to the 'left'
             
             But for the next row of patches, we need to check overlap with 2 sides:
                patch 'above' and patch to the 'left'
        
            '''
            for i in range(patch_size):
                if(dst_i + i == max_i):
                    break
                for j in range(patch_size):
                    if (dst_j + j == max_j):
                        break
                        
                    out_img_r[dst_i+i][dst_j+j] = sample[i][j][0]
                    out_img_g[dst_i+i][dst_j+j] = sample[i][j][1]
                    out_img_b[dst_i+i][dst_j+j] = sample[i][j][2]
                              
                #print('quilt_simple:: dst_i:', str(dst_i), ' i: ', str(i), \
                #  ' dst_j: ', str(dst_j),' j: ', str(j))
            
            dst_j += patch_size
            
            if (dst_j >= max_j):
                dst_j = 0
                break

            sample = select_sample_seamcut(samples, prev_patch, patch_size, overlap, tol)   
            prev_patch = sample
           
            num_patches += 1
            
        dst_i += (patch_size)
            
        if (dst_i >= max_i):
            break
       
    print('Number of patches utilized in generating image: ', str(num_patches))
    
    out_img_bgr = cv2.merge((out_img_b, out_img_g, out_img_r))
    out_img_rgb = cv2.merge((out_img_r, out_img_g, out_img_b))

    return (out_img_bgr, out_img_rgb)
    

def resize_image(source_img, target_img):

    print('sample shape: ', source_img.shape)
    print('target shape: ', target_img.shape)
    
    h1, w1, c1 = source_img.shape
    h2, w2, c2 = target_img.shape
    
    new_img = cv2.resize(source_img, (w2,h2))
    #new_img2 = quilt_cut(source_img, max(h2,w2), int(max(h2,w2)/5), 10, 0.001)
    
    return new_img

