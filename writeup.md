# Vehicle Detection Project

## Solution based on YOLO method

I have used alternative method for this project, based on YOLO, repository is [here](https://github.com/kbobrowski/YOLO-vehicle-detection). Video with a final result:

[![video](https://img.youtube.com/vi/64bETGQ-tLk/0.jpg)](https://www.youtube.com/watch?v=64bETGQ-tLk)

## Solution based on a classic approach

This repository and writeup file contains classic approach introduced in the lecture.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `lesson_functions.py`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and optimized them manually for maximum classifier accuracy.

Final values are:

```python
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `vehicles` and `non-vehicles` database from Udacity. Spatial features, histogram features and HOG features has been combined in a features vector. Classifier has been trained using `LinearSVC` (line 83 in `train.py`)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented in a `slide_window` function (`lesson_functions.py`, line 100). I have used two sizes of sliding window: 64x64 and 96x96, with 0.5 overlapping, which could accurately determine car position.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here is an example:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Furthermore, frame for each car is labeled and stabilized using Tracker class (`Tracker.py`). It attempts to stabilize both movement and size change of a frame, as well as preserve frame if there is short break in a detection. It also eliminates false positives by drawing a frame only after it is visible for set time. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Main issue was poor detection quality using sliding window. It was reduced by training classifier on a larger dataset, as well as by introducing two scales for a sliding window. A pipeline fails when two cars are close to each other (they are identified as one). It is caused by a heatmaps from both cars merging into one. The effect could be reduced by introducing more sliding window scales and smaller overlap, combined with tuning the threshold. Alternatively, YOLO approach can be used, which by default can correctly distinguish two slightly overlapping cars. This, combined with methods implemented in `Tracker.py` class provides better performance.
