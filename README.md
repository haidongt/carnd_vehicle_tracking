**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./for_report/test3_output.jpg "image 1"
[image2]: ./for_report/test3_output1.jpg "image 2"
[image3]: ./for_report/test6_output.jpg "image 3"
[image4]: ./for_report/test6_output1.jpg "image 4"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted features from the training images.

The code for this step is contained train_svm.py. I used three sets of features: color space features, color histogram features, hog features. In order to search hyper parameters automatically, I used GridSearchCV from sklearn.model_selection. I defined several transformers for my pipeline. For color space features, I defined a ColorSpaceConverter that transforms RGB color space to other specified color spaces and then I defined a SpatialBining transformer to generate the features. For color histogram features and hog features I also applied ColorSpaceConverter before generating the features. All features are then normalized before they are combined.

####2. Explain how you settled on your final choice of HOG parameters.

As I mentioned above, I used GridSearchCV from sklearn.model_selection to automatically find the best parameters. I first define a set of parameters that I want to experiment with and then let the library find the best parameters for me. Although it also takes a lot of time to run, luckily I only have to do it once and the result is satisfactory. I had a test accuracy over 99%.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code is in train_svm.py. I defined a pipeline that first generate three sets of features: color space features, color histogram features, hog features, and I then use sklearn.pipeline.FeatureUnion to combine these features. Afterwards, I feed the combined features into sklearn.svm.LinearSVC to train a SVM model.

####4. Final selection of parameters
These are the parameters that I used for my final model:
HOG number of orientation: 18
HOG cells per block: 2
HOG Pixel per cell: 8
Color space for extracting HOG features: LAB
Color space for Spacial Binning: LAB
Color histogram bins: 32
Color space for color histogram: HLS
Spatial binning bins: 32

I used GridSearchCV to automatic select the parameters. I inspect the parameters for the best model by calling
```print('Best params: ', cls.best_params_)```
and I got printout of "Best params:  {'orient': 18, 'cells_per_block': 2, 'cs_cspace': 'LAB', 'chist_bins': 32, 'pix_per_cell': 8, 'chist_cspace': 'HLS', 'cs_bins': 32, 'hog_cspace': 'LAB'}"

I then save the model to svm_model.p and load the same model in the later video pipeline.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is in image_utils.py VehicleDetector.get_windows() and VehicleDetector.search_windows(). I used a fixed set of windows. These windows are generated manually and they contain three different sizes. The largest windows cover the bottom of the image and the smallest windows cover the middle portion of the image. For each frame I itererate through these windows and resize the part of image to 64 * 64 and then apply the classifier that I trained to determine if it's a car window or no-car window.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I used HLS space for color space features and used 32 bins for color histogram features and used 18 orientations for hog features.
Here are some example images:

![alt text][image2]
![alt text][image1]
![alt text][image4]
![alt text][image3]

####3. How I improved the reliability of the classifier.
I used SVM's decision function to determine how confident the classifier is with the prediction. I set a threshold to filter the positives with low confidence. I found that setting a threshold of 0.5 is efficient of filtering out false positives.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
[Link](https://www.youtube.com/watch?v=oVmQzNVMxDc)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap over the last 10 frames and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that manually tuning the hyper parameters for the SVM is very time consuming so I used GridSearchCV from sklearn.model_selection to automatically find the best hyper parameters. Since the heatmap is generated over 10 consecutive frames, it assumes that vehicle's position in the image do not change much over the 10 frames. However, if there's a very fast vehicle that changes its location greatly over 10 frames. The pipeline is likely to fail. Also the pipeline does not distinguish individual vehicles when multiple vehicles are partically overlapped. To make the pipeline more robust, after identify the rough locations of vehicles, a more detailed algorithem could be designed to distinguish vehicles and generate tighter bounding boxes for multiple vehicles. For example, color features can be used again to distinguish different partially overlapped vehicles.
