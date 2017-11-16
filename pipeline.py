import matplotlib.image as mpimg
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from skvideo.io import FFmpegWriter
from scipy.ndimage.measurements import label
from Tracker import Tracker


color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()

import pickle
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
image = mpimg.imread("testgit.jpg")
image = image.astype(np.float32)/255
draw_image = np.copy(image)

video = "project_video.mp4"
videoOut = "boxes.mp4"

vidcap = cv2.VideoCapture(video)
fcount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
success, img = vidcap.read()
shape = img.shape
vidout = FFmpegWriter(videoOut)
if not success: sys.exit(1)
i = 0
fps_avg = []


tracker = Tracker()

while True:
    i+=1
    img_converted = convert_cv_mpl(img)
    draw_image = np.copy(img)
    t1 = time.time()
    windows = slide_window(img_converted, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_windows=[(64,64),(96,96)], xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(img_converted, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    #img_plot = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    heat = np.zeros_like(draw_image[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 0.5)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    #img_plot = draw_labeled_bboxes(np.copy(draw_image), labels, draw=True)
    boxes = draw_labeled_bboxes(np.copy(draw_image), labels, draw=False)
    for box in boxes:
        xmin = box[0][0]
        xmax = box[1][0]
        ymin = box[0][1]
        ymax = box[1][1]
        centerx = (xmin + xmax)/2
        centery = (ymin + ymax)/2
        frameimg = draw_image[ymin:ymax, xmin:xmax]
        tracker.new_object([centerx, centery], frameimg)
    img_plot = np.copy(draw_image)
    tracker.draw_frames(img_plot)
    
    t2 = time.time()
    fps = 1/(t2-t1)
    fps_avg.append(fps)
    if len(fps_avg) > 200: fps_avg.pop(0)
    fps_print = int(np.mean(fps_avg))
    #cv2.putText(img_plot, "FPS: {}".format(fps_print), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    vidout.writeFrame((convert_cv_mpl(img_plot)*255).astype(np.uint8))
    cv2.imshow('img', img_plot)
    #printProgressBar(i, fcount, "Progress:", "({} FPS)".format(fps_print), length=20)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    success, img = vidcap.read()
    if not success: break

vidcap.release()
vidout.close()
cv2.destroyAllWindows()
