# **Traffic Sign Recognition** 

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

[image1]: ./plots/class_occurence_bar.png "Occurence of Classes"
[image2]: ./plots/grayscaled.png "Grayscaling"
[image3]: ./plots/normalized.png "Normalization"
[image4]: ./plots/class_grid.png "Examples"
[image5]: ./plots/sign1_cropping.png "Traffic Sign 1"
[image6]: ./plots/sign2_cropping.png "Traffic Sign 2"
[image7]: ./plots/sign3_cropping.png "Traffic Sign 3"
[image8]: ./plots/sign4_cropping.png "Traffic Sign 4"
[image9]: ./plots/sign5_cropping.png "Traffic Sign 5"
[image10]: ./plots/conv1_feature_map.png "Feature Map"
[image11]: ./plots/predictions.png "Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/alexramos/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 samples.
* The size of the validation set is 4,410 samples.
* The size of test set is 12,630 samples.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

##### Figure 1.
German Traffic Sign data, example for each class.

![alt text][image4]

##### Figure 2.
Grouped bar chart showing the percentage of each class across the three train, validation, and test datasets.  This shows that the distribution of classes are similar across datasets. One notable difference is that the validation set has slightly more representation of labels [0, 6, 15, 19, 20, 21, 22, 30, 33, 34, 36, 40].

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the images to grayscale because to reduce noise in the images and reduce the amount of data used by the network.  Additionally, this often makes traffic signs more apparent in darker images.  See the example below.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to increases the numerical stability of our computations and allow gradient descent to perform quicker.  Images are normalized so that pixel values have a mean of zero and equal variance using (pixel - 128)/ 128.  Implemented

Here is an example of a grayscaled traffic sign image before and after normalization.

![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x128  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 				|
| Fully connected		| input = 3200, output = 120 |
| RELU					|												|
| Drop-out					|												|
| Fully connected		| input = 120, output = 84 |
| RELU					|												|
| Drop-out					|												|
| Fully connected		| input = 84, output = 43 |
|			SoftMax			|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I mostly used the LeNet pipline from the previous project, but with some minor modifications.  I used the AdamOptimizer to minimize a loss function with batch size of 128 and learning rate of 0.001.  I trained for 20 epochs and a keep probability of 0.5 for drop-out during training

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My approach consisted of using the LeNet architecture with two major modifications:

1. I increased the filter depth of the first two convulation layers to 48 and 128, respectively.  This had a significant impact on peformance, leading to an 3% increase in accuracy on the validation set.

2. I implemented drop-out after each fully-connected activation layer.  This also had a significant impact (+3%) on validation set accuracy.

We settled on these two modifications after iteratively assessing various network architectures.

My final model results were:
* training set accuracy of 0.999 (Section "Train the Model", 16th cell of notebook)
* validation set accuracy of 0.972 (Section "Train the Model", 16th cell of notebook)
* test set accuracy of 0.959 (Section "Evaluate the Model", 17th cell of notebook)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture that was tried was the unaltered LeNet approached used for classifying the MNIST data.

* What were some problems with the initial architecture?

The initial architecture did not perform well with our validation set, with an accuracy of 0.89.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I started out by increasing the filter depth of the first two convolution layers.  This led to improved performance on both the test and validation set.  I then experimented with adding additional convolution layers and increasing the number of neurons in the fully-connected layers.

* Which parameters were tuned? How were they adjusted and why?

I increased the number of epochs used for training by 100% to 20.  With GPU-based computation, this had a minimal impact on training time while leading to an increased final validation set accuracy.  I also experimented by decreasing the learning rate to decrease training time but this led to poorer performance.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A ConvNet is ideal for classifying images as it mathematically models the hiearchical pattern detection that occurs naturally in our visual cortex.  The architecture used here employs two convolution layers with large filter depths of 48 and 128, respectively.  I also added drop-out after each fully-connected ReLu activation layer.  The reduces overfitting and leads to notable increases in validation set accuracy.

If a well known architecture was chosen:
* What architecture was chosen?

LeNet architecture was used for this project.

* Why did you believe it would be relevant to the traffic sign application?

I believe LeNet is an appropriate choice of network architecture because it's been shown to perform exceptionally well on image classification of single digit numbers (MNIST dataset).  The German Traffic Sign dataset is more complex than MNIST yet LeNet still performs very well overall.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
The final model's accuracy of 0.999, 0.972, and 0.959 on training, validation, and test sets indicate the model is a bit overfitted to the training set and there is room for improvement.  Nonetheless, the model is performing much better than the initial performance of the unaltered LeNet architecture.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The first image, a 30 km/h sign might be difficult to classify because it is within a larger rectangular shape and it's similar to the 80km/h sign.  The second image, a keep right sign, could prove difficult to classify because it is padded on the left and right sides of the image.  The third image, a circular "No vehicles" sign, is posted above another rectangular sign, which might confuse the classifier.  The fourth image, diamond-shape priority road sign, should be  straightforward to classify.  Lastly, the fifth image, a general caution sign, is similar to the third image, and is also posted above a rectangular sign with wording.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   									| 
| Keep right     			| Keep right 										|
| No vehicles					| No vehicles											|
| Priority road	      		| Priority road					 				|
| General Caution			| General Caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.9%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image (Speed limit (30km/h)), the model is very sure that this is a 30 km/h speed limit sign (probability of 1.0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)   									| 
| 3e-12     				| Speed limit (20km/h) 										|
| 1e-12					| Speed limit (50km/h)											|
| 3e-14	      			| Speed limit (80km/h)					 				|
| 2e-17				    | End of speed limit (80km/h)      							|

For the second image (Keep right), the model is very sure that this is a Keep right sign (probability of 1.0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep right   									| 
| 0.     				| Speed limit (20km/h) 										|
| 0.					| Speed limit (30km/h)											|
| 0.	      			| Speed limit (50km/h)					 				|
| 0.				    | Speed limit (60km/h)      							|

For the third image (No vehicles), the model is very sure that this is a No vehicles sign (probability of 1.0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No vehicles   									| 
| 2e-13     				| Speed limit (120km/h) 										|
| 4e-15					| Priority road											|
| 1e-17	      			| Yield					 				|
| 4e-19				    | No passing      							|

For the fourth image (Priority road), the model is very sure that this is a Priority road sign (probability of 1.0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road   									| 
| 3e-24     				| Roundabout mandatory 										|
| 8e-25					| No vehicles											|
| 2e-31	      			| Turn right ahead					 				|
| 1e-31				    | End of no passing by vehicles over 3.5 metric tons      							|

For the fifth image (General caution), the model is very sure that this is a General caution sign (probability of 1.0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999         			| General caution   									| 
| 3e-7     				| Right-of-way at the next intersection 										|
| 1e-9					| Pedestrians											|
| 6e-10	      			| Speed limit (30 km/h)					 				|
| 1e-10				    | Speed limit (50 km/h)      							|

Below are bar charts visualizing the soft-max probabilities for each of the 5 images tests.
![alt text][image11]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below are the feature maps used by the first convolution layer of my network after running on a Keep right sign from the web.  You can see that the feature maps in this first layer are recognizing both the circular lines and diagnal lines in the image.

![alt text][image10]
