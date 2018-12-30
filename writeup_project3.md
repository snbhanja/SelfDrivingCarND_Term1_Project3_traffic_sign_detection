# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/01_eda.png "eda"
[image2]: ./report_images/02_class_distribution.png     "class_distribution"
[image3]: ./report_images/03_undistorted.png "undistorted"
[image4]: ./report_images/04_distorted.png "distorted"
[image5]: ./report_images/05_random_noise.png "random_noise"
[image6]: ./report_images/06_invert_color.png "invert_color"
[image7]: ./report_images/07_horizontal_flip.png "horizontal_flip"
[image8]: ./report_images/08_blurred_image.png "blurred_image"
[image9]: ./report_images/09_model_architecture.jpg "model_architecture"
[image10]: ./report_images/10_lenet.jpeg "model architecture diagram"


[image11]: ./new_test_images/end-of-all-speed-limits.jpg  
[image12]: ./new_test_images/no-passing-2.jpg  
[image13]: ./new_test_images/road-work.jpg  
[image14]: ./new_test_images/turn-right-ahead.jpg "model architecture diagram"
[image15]: ./new_test_images/dangerous-turn-to-the-left.jpg
[image16]: ./report_images/11_5images_prediction.png

[image17]: ./report_images/12_prediction_1.JPG
[image18]: ./report_images/13_prediction_2.JPG
[image19]: ./report_images/14_prediction_3.JPG
[image20]: ./report_images/15_prediction_4.JPG
[image21]: ./report_images/16_prediction_5.JPG
[image22]: ./report_images/17_cnn1.png
[image23]: ./report_images/18_cnn2.png
[image24]: ./report_images/19_cnn3.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
The code for this step is contained in the third code cell of the IPython notebook.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
The code for this step is contained in the fourth code cell of the IPython notebook. <br/>
![alt text][image1]<br/>

It is a bar chart showing how the data. See code cell 6....<br/>

![alt text][image2]<br/>

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The code for this step is contained in the 7th and 8th cells of the IPython notebook.

My preprocessing pipeline consists of the following steps:

1. Conversion to grayscale: It didn't significantly change the accuracy, but it made it easier to do the normalization.
2. Saturating the intensity values at 1 and 99 percentile.
3. Min/max normalization to the [0, 1] range.
4. Subtraction of its mean from the image, making the values centered around 0 and in the [-1, 1] range.
5. Apply random noise.
6. Invert the color of the images.
7. Horizontal flipping of images
8. Blurring of images.

The undistorted image as below, <br/>
![alt text][image3] <br/>

After applying steps 1 to 4 above, the distorted image as below, <br/>
![alt text][image4] <br/>

Random noise: <br/>
![alt text][image5] <br/>

Invert color: <br/>
![alt text][image6] <br/>

Horizontal flip: <br/>
![alt text][image7]  <br/>

Blurred image: <br/>
![alt text][image8] <br/>


With a simple min/max normalization I had (approx.) 1% better validation-accuracies than without it. The percentile-based method gave an additional (approx.) 1% improvement over simple min/max normalization (this method was mentioned in [3]). 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 14th cell of the ipython notebook.

My architecture is a deep convolutional neural network inspired by two existing architectures: one is LeNet[1], and the other is the one in Ciresan's paper[3]. Its number and types of layers come from LeNet, but the relatively huge number of filters in convolutional layers came for Ciresan. Another important property of Ciresan's network is that it is multi-column, but my network contains only a single column. It makes it a little less accurate, but the training and predition is much faster.

The model architecture diagram as below, <br/>
![alt text][image10] <br/>

The final model consisted of the following layers: <br/>
![alt text][image9] <br/>


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 15th to 19th cell of the ipython notebook.

To train the model, I used AdamOptimizer, a batch size of 128, at most 30 epochs, a learn rate of 0.001. Another hyperparameter was the dropout rate which was 0.7 at every place where I used it. I have tried changing these parameters but it didn't really increase the accuracy. I saved the model which had the best validation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twelfth cell of the Ipython notebook.

My final model results were:

training set accuracy of 0.991
validation set accuracy of 0.980
test set accuracy of 0.961

I started out by creating an architecture which could clearly overfit the training data. (It converged to 1.0 training-accuracy in a couple of epochs, but the validation accuracy was much lower. Then I have added regulators until the overfitting was more-or-less eliminated. I added dropout operations between the fully connected layers. I also tried L2 regularization for the weights (in addition to the dropout), but it made the accuracy worse by a tiny amount. Then I have kept removing filters up to the point when the accuracy started decreasing.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web: <br/>

![alt text][image11] <br/>
![alt text][image12] <br/>

This picture is interesting because the perspective and rotation makes the car figures almost form a diagonal similar to the one in End of all speed limits sign. <br/>

![alt text][image13] <br/>
![alt text][image14] <br/>

This picture might be hard to classify because it is from a strange perspective and another sign is hanging in to the picture. <br/>

![alt text][image15] <br/>

This is a very blurry image, at first even I didn't know which sign is it. It might be a photo or an artistic painting, of most likely the "Dangerous turn to the left" sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image16] <br/>

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.
This is lower than the test accuracy of 96.1%, however I would not draw conclusions from this very small (5 images) dataset.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

For the first image, the model is very confident (99.2%), and correct. The top five soft max probabilities were:<br/>
![alt text][image17] <br/><br/>

For the second image, the model is relatively sure (27%) that we are looking at a "Dangerous curve to the right " sign , but the correct answer is No passing which was not even in the top five guesses. Maybe the image was too skewed.<br/>
![alt text][image18] <br/><br/>

For the third image, the model is completely confident (100%), and correct.<br/>
![alt text][image19] <br/><br/>

For the fourth image, the model is relatively sure (18%), and correct.<br/>
![alt text][image20] <br/><br/>

For the fifth image, the model is very sure (95.16%) that it is a Bumpy road sign, but the correct answer is Dangerous curve to the left, which was not even in the top five guesses.<br/>
![alt text][image21] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In the visualization of the activations of the first convolutional layer, I have seen that different feature maps look for different edges, for example the 7th feature map is activated by diagonal edges, and the 8th feature map is activated by horizontal edges.

![alt text][image22] <br/><br/>
![alt text][image23] <br/><br/>
![alt text][image24] <br/><br/>

Reference<br/>
[1] Lecun(1998): Gradient-Based Learning Applied to Document Recognition

[2] Sermanet(2011): Traffic Sign Recognition with Multi-Scale Convolutional Networks

[3] Ciresan (2012): Multi-Column Deep Neural Network for Traffic Sign Classification

[4] https://www.kaggle.com/tomahim/image-manipulation-augmentation-with-skimage

