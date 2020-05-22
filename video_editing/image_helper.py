import os
import cv2
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt

def read_image(im_file):
    if (not os.path.exists(im_file)):
        print('File ', im_file, ' does not exist!')
        return None

    return cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

def write_image(im_path, im, dtype=np.uint8):
    bgr_image = (im[:, :, [2, 1, 0]]).astype(dtype)
    if (os.path.exists(im_path)):
        os.remove(im_path)

    return cv2.imwrite(im_path, bgr_image)

def map_image_to_canvas(im, canvas):
    canvas_1 = np.zeros(canvas.shape, canvas.dtype)
    H,W,C = canvas.shape
    im_h,im_w,im_c = im.shape
    y_start = int(H/2)-int(im_h/2)
    x_start = int(W/2)-int(im_w/2)
 
    for y in range(im_h):
        for x in range(im_w):
            for c in range(im_c):
                canvas_1[y+y_start][x+x_start][c] = im[y][x][c]

    #plt.imshow(canvas_1)
    return canvas_1

def trim_image(im):
    
    H,W,C = im.shape
    
    last_blank = 0
    
    start_y = -1
    end_y = -1

    im_T = np.transpose(im) # dim = C, W, H
    
    #print('im.shape: ', im.shape, ' im_T.shape: ', im_T.shape)
        
    sum_rows = np.ndarray.sum(im_T[0], axis=0)
    sum_cols = np.ndarray.sum(im_T[0], axis=1)

    sum_rows_1 = np.ndarray.sum(im_T[1], axis=0)
    sum_cols_1 = np.ndarray.sum(im_T[1], axis=1)
    
    sum_rows_2 = np.ndarray.sum(im_T[2], axis=0)
    sum_cols_2 = np.ndarray.sum(im_T[2], axis=1)
    
    #print('sum_rows: ', sum_rows.shape)
    #print('sum_cols: ', sum_cols.shape)

    for y in range(H):
        if (sum_rows[y] == 0 and sum_rows_1[y] == 0 and sum_rows_2[y] == 0):
            if (start_y != -1):
                #print("Blending region startpoint recognised")
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
        end_y = H
       
    #print("start_y: ", start_y, " end_y: ", end_y)  
    
    last_blank = 0
    start_x = -1
    end_x = -1
    
    for x in range(W):
        if (sum_cols[x] == 0 and sum_cols_1[x] == 0 and sum_cols_2[x] == 0):
            if (start_x != -1):
                #print("Blending region startpoint recognised")
                if(last_blank + 1 == x):
                    end_x = x
                    #print("Last non blank col:", x)
                    break
            last_blank = x
        else:
            if (start_x == -1):
                #print("First non blank col:", x)
                start_x = x
        
    if(start_x == -1):
        start_x = 0
    if (end_x == -1):
        end_x = W
        
    #print("start_x: ", start_x, " end_x: ", end_x)
    
    new_h = end_y - start_y + 1
    new_w = end_x - start_x + 1
    
    new_im = np.zeros((new_h, new_w, C), dtype=im.dtype)
    
    #print("New size: ", new_h, " X", new_w, " Shape: ", new_im.shape, " Size: ", new_im.size)
    
    y = start_y
    new_y = 0
    
    while(y < end_y):
        x = start_x
        new_x = 0
        while (x < end_x):
            for c in range(C):
                new_im[new_y][new_x][c] = im[y][x][c]
            x += 1
            new_x += 1
            
        y += 1
        new_y += 1
    
    return new_im

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def specify_bottom_center(img):
    print("If it doesn't get you to the drawing mode, then rerun this function again. Also, make sure the object fill fit into the background image. Otherwise it will crash")
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    fig.set_label('Choose target bottom-center location')
    plt.axis('off')
    target_loc = np.zeros(2, dtype=int)

    def on_mouse_pressed(event):
        target_loc[0] = int(event.xdata)
        target_loc[1] = int(event.ydata)
        
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    return target_loc

def align_source(object_img, mask, background_img, bottom_center):
    ys, xs = np.where(mask == 1)
    (h,w,_) = object_img.shape
    y1 = x1 = 0
    y2, x2 = h, w
    object_img2 = np.zeros(background_img.shape)
    yind = np.arange(y1,y2)
    yind2 = yind - int(max(ys)) + bottom_center[1]
    xind = np.arange(x1,x2)
    xind2 = xind - int(round(np.mean(xs))) + bottom_center[0]

    ys = ys - int(max(ys)) + bottom_center[1]
    xs = xs - int(round(np.mean(xs))) + bottom_center[0]
    mask2 = np.zeros(background_img.shape[:2], dtype=bool)
    for i in range(len(xs)):
        mask2[int(ys[i]), int(xs[i])] = True
    for i in range(len(yind)):
        for j in range(len(xind)):
            object_img2[yind2[i], xind2[j], :] = object_img[yind[i], xind[j], :]
    mask3 = np.zeros([mask2.shape[0], mask2.shape[1], 3])
    for i in range(3):
        mask3[:,:, i] = mask2
    background_img  = object_img2 * mask3 + (1-mask3) * background_img
    plt.figure()
    plt.imshow(background_img)
    return object_img2, mask2


def specify_keypoints(img):
    # get mask
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    fig = plt.figure()
    fig.set_label('Draw polygon around source object')
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    clicked = []

    def on_mouse_pressed(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        plt.plot(x, y, 'r+')

    def onclose(event):
        clicked.append(xs)
        clicked.append(ys)
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    fig.canvas.mpl_connect('close_event', onclose)
    return clicked

def specify_mask(img):
    # get mask
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    fig = plt.figure()
    fig.set_label('Draw polygon around source object')
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    clicked = []

    def on_mouse_pressed(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        plt.plot(x, y, 'r+')

    def onclose(event):
        clicked.append(xs)
        clicked.append(ys)
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    fig.canvas.mpl_connect('close_event', onclose)
    return clicked

def get_mask(ys, xs, img):
    mask = poly2mask(ys, xs, img.shape[:2]).astype(int)
    fig = plt.figure()
    plt.imshow(mask, cmap='gray')
    return mask


def specify_bottom_center(img):
    print("If it doesn't get you to the drawing mode, then rerun this function again. Also, make sure the object fill fit into the background image. Otherwise it will crash")
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    fig.set_label('Choose target bottom-center location')
    plt.axis('off')
    target_loc = np.zeros(2, dtype=int)

    def on_mouse_pressed(event):
        target_loc[0] = int(event.xdata)
        target_loc[1] = int(event.ydata)
        
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    return target_loc

