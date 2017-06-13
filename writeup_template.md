##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier  -- COMPLETED
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.  -- COMPLETED
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.  -- COMPLETED
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.  -- COMPLETED
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.  -- COMPETED, UPDATED WITH REVIEW COMMENTS (Add multiple windows size, combined heatmap for continuous frames)
* Estimate a bounding box for vehicles detected.  -- COMPLETED

[//]: # (Image References)
[image1]: .h.png
[image2]: ./hog_feature.png
[image3]: ./hog_feature_ycrcb.png
[image4]: ./sliding_window.png
[image5]: ./heatmap.png
[image6]: ./labels_and_box.png
[image7]: ./last_frame.png
[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Here are my steps of extracting HOG features from the training images:
* Unzip the vehicle.zip and non-vehicle.zip.   There are about 9000 images on each zip file.
* create a fuction and use the skimage hog function to get the feature and hog_image.   
* to add on the binned color feature and color histogram features to define an extract feature.  In the extract_feature, I have option for bin_spatial on/off, histogram on/off and hog_feature on/off.
* The order of the feature is binned color feature, then color histogram and then hog features


![alt text][image1]
I have the following setting for the extraction
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

In the later on process, I found out I get higher score on training with color_space = 'YCrCb' and hog_channel = 'ALL'.   The hog image is not all 3 channel, I just pick the channel 2 in this case.

![alt text][image2]


#### 2 . Explain how you settled on your final choice of HOG parameters.

It is very helpful to train the data with smaller set first to see score and then train all the data.   It tooks about 90s to do feature extraction for 18,000 images.

I use the following setting for final result.   I have tried RGB and hog_channel = 0.   It give me about 95% score.  I switched to LUV and YUV, it give me about 96%.  Change to use all Hog channel.   It gave to 98%.  Changed color_space to YCrCb and use all hog_channel.  The percent is 99.38% which is pretty good.  For the rest of the parameters, I kept them as in the lesson.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For both car features and not car features extraction.  I modify the parameter couple time to see the best score.   I finally use YCrCb and all hog channels to get 99.38% score. 

I splitted the train data and test data by 80% and 20%.   Randomized the data before splitting.

The extraction took 90s but the fitiing only took 3.61 second.   The accurary is 99.38%.    The default parameter from the lesson is about 95% with RGB and single hog channel.  


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with no y-start = None, too many false positive.   I put a starting point at 400 and then y-stop at (n * windows_size + y-stop) which is close to 728.   For example, if the windows is 96x96, teh y-stop is 3 * 96 + 400 = 688.   

I tried the windows size 128x128 and 64x64.  On 128x128, I miss the car in some image, and on 64x64, I see more false positive.   I use 96x96.   In every image, there are 125 windows.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()


I have played with the color space and hog channel, with YCrCb and Hog channel = ALL, I get 99.32%
105.56605982780457  seconds to complete features
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
19.14 Seconds to train SVC...
Test Accuracy of SVC =  0.9932


---

### Video Implementation

#### 1. Run the pipeline result.  Instead of t
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

Based on the first review feedback:
I added 2 windows size search in different areas:
    ystart1 = 400, ystop1 = 500, scale1 = 0.8   -- closer to horizon, 64 x 0.8
    ystart2 = 400, ystop2 = 688, scale2 = 1.5   -- second half, windows size = 64 x 1.5 = 96

    deque length = 5, which add up the heatmap for 5 frames, threshold >= 2 

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There are couple issues I have seen
* The car that is too far away are not detected.   I tried to decrease y start point, it does not help.   I probably have to reduce the windows size to get those smaller car.  However, I did scale down the windows size, it gave more false positive.   I set the threshold = 0, it helps but give more false position in other place.
-- Resolved by adding multiple car_find and merge heatmap.   

* I detect the car on the opposite lane.  Couple people in the forum said it is a good thing.   It may be useful in some case as long as I can recongize it coming from oppositin direction.   I can see it helpful when I need the info for passing a single lane and avoid head-to-head collision.




