import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_images(img_cnt, imgs, titles=None):
    
    assert img_cnt == len(imgs)
    if (titles is not None):
    	assert img_cnt == len(titles)
    else:
    	titles = []
    	for cnt in range(img_cnt):
            title_str = str('Img{}'.format(cnt+1))
            titles.append(title_str)
        
    fig, axes = plt.subplots(1,img_cnt, figsize=(25,25))

    for i in range(img_cnt):
        axes[i].imshow(imgs[i],cmap='gray') 
        axes[i].set_title(titles[i], fontsize=15)

def blend_images(sourceTransform, referenceTransform):
    '''
    Naive blending for frame stitching
    Input:
        - sourceTransform: source frame projected onto reference frame plane
        - referenceTransform: reference frame projected onto same space

    Output:
        - blendedOutput: naive blending result from frame stitching
    '''

    blendedOutput = referenceTransform.copy()
    sourceImage = sourceTransform.copy()
    indices = referenceTransform == 0
    blendedOutput[indices] = sourceImage[indices]

    return (blendedOutput / blendedOutput.max() * 255).astype(np.uint8)    

def get_img_corners(img):
    '''
    Returns the position of the corners of an color image
    Input:
        img: color image.
    Output:
        corners: image corners (np.array)
    '''	
    corners = np.zeros((4, 1, 2), dtype=np.float32)

    height, width, channels = img.shape
    corners[0] = (0, 0) #top left
    corners[1] = (width, 0) #top right
    corners[2] = (width, height) #bottom right
    corners[3] = (0, height) #bottom left
    
    return corners    

def gamma_correction(im, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    im_enh = cv2.LUT(im, lookUpTable)
    
    return im_enh

def video2imageFolder(input_file, output_path):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_file: Input video file.
        output_path: Output directorys.
    Output:
        None
    '''

    cap = cv2.VideoCapture()
    cap.open(input_file)

    if not cap.isOpened():
        print("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Frame count: {}'.format(frame_count))

    frame_idx = 0
    file_idx = 0
    while frame_idx < frame_count:
        ret, frame = cap.read()

        if not ret:
            print ("Failed to get the frame {}".format(frame_idx))
            frame_idx += 1
            continue

        out_name = os.path.join(output_path, 'f{:04d}.jpg'.format(file_idx+1))

        if (not os.path.exists(out_name)):
            ret = cv2.imwrite(out_name, frame)
            if not ret:
                print ("Failed to write the frame {}".format(frame_idx))
                frame_idx += 1
                continue

        frame_idx += 1
        file_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def imageFolder2mpeg(input_path, output_path='./output_video.mpeg', fps=30.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_path: Input video file.
        output_path: Output directorys.
        fps: frames per second (default: 30).
    Output:
        None
    '''

    dir_frames = input_path
    files_info = os.scandir(dir_frames)

    file_names = [f.path for f in files_info if f.name.endswith(".jpg")]
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame_Height, frame_Width = cv2.imread(file_names[0]).shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MPG1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    frame_count = len(file_names)

    frame_idx = 0

    while frame_idx < frame_count:


        frame_i = cv2.imread(file_names[frame_idx])
        video_writer.write(frame_i)
        frame_idx += 1

    video_writer.release()

def specify_keypoints(img):
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
