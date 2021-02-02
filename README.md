# Traffic Sign Recognition

Overview
---

The project implemented as:
- [Jypytier Notebook](./Traffic_Sign_Classifier.ipynb)
- [HTML report](./report.html)
- [PDF report](./report.pdf)

In this project, I'll use what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I'll will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[augmented]: ./output/augmented.png "Augmented Image"
[pre_processed]: ./output/download.png "Preprocessed Images"
[examples]: ./output/examples.png "Examples"
[lc]: ./output/learning_curve.png "Learning Curve"
[n_signs]: ./output/number_of_signs.png "Distribution"

---

## Data Set Summary & Exploration

Generic overview of input data:

* The training set size is 34799
* The validation set size is 4410
* The test set size is 12630
* The shape of an image is (32, 32, 3)
* There're 43 unique classes


### Distribution:

![alt text][n_signs]

Most common signs:
* `Speed limit (50km/h)`  train samples: 2010
* `Speed limit (30km/h)`  train samples: 1980
* `Yield`  train samples: 1920
* `Priority road`  train samples: 1890
* `Keep right`  train samples: 1860


Most rare signs:

* `Speed limit (20km/h)`  train samples: 180
* `Dangerous curve to the left`  train samples: 180
* `Go straight or left`  train samples: 180
* `Pedestrians`  train samples: 210
* `End of all speed and passing limits`  train samples: 210

### Examples of signs:

![alt text][examples]

---

## Design and Test a Model Architecture

### Preprocessing

I applied normalization similar to the [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Images were transformed in the YUV space and adjusted by
histogram sketching and by increasing sharpness. For the next step only Y channel was selected.

All images were additionally augmented to get more test samples. Here is an example of an original image and the augmented image.

![alt text][augmented]

### The network architecture

The model consists of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 image   				    |
| Convolution    	| Filter 5x5, 1x1 stride, valid padding, outputs 28x28x6 	|
| Relu					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution	    | Filter 5x5, 1x1 stride, valid padding, outputs 10x10x16    |
| Relu					|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|  			|
| Fully connected		| Input 5x5x16 output 2*400       			|
| Dropout					|												|
| Fully connected		| Input 800 output 120       			|
| Relu					|												|
| Fully connected		| Input 120 output 84       			|
| Relu					|												|
| Fully connected		| Input 84 output 43       			|
| Softmax				|												|
|						|												|

I trained the model using the Adam optimizer (with momentum),  the learning rate is 1e-4 , dropout rate of 0.2 and batch size of 128.

##### Answer:

To train the model, I started from the LeNet.
After a few runs with this architecture I noted that the model tended to overfit. Then I added the dropout optimization and changed a little bit the architecture of the layers. Initially, I used a high dropout rate 75%, but later I switched to 20%.

I used 200 epochs. Below you can find the learning curve:

![alt text][lc]

Final result:
* Accuracy on training images: 97%
* Accuracy on validation images: 93%
* Accuracy on validation images: 91%

---
## Test a Model on New Images

I found a few images on the Web and I preprocessed them:

![alt text][pre_processed]

These images:

* Include harder background
* Contain artifacts such as jpeg compression

Results of the prediction:

Top 5 Labels for image 'Stop':
 - 'Stop' with prob = 1.00
 - 'No entry' with prob = 0.00
 - 'Priority road' with prob = 0.00
 - 'Road work' with prob = 0.00
 - 'Turn left ahead' with prob = 0.00

Top 5 Labels for image 'Speed limit (70km/h)':
 - 'Speed limit (70km/h)' with prob = 0.55
 - 'Speed limit (20km/h)' with prob = 0.45
 - 'Speed limit (30km/h)' with prob = 0.00
 - 'Keep right' with prob = 0.00
 - 'Speed limit (120km/h)' with prob = 0.00

Top 5 Labels for image 'Priority road':
 - 'Priority road' with prob = 1.00
 - 'No vehicles' with prob = 0.00
 - 'Road work' with prob = 0.00
 - 'No passing for vehicles over 3.5 metric tons' with prob = 0.00
 - 'No passing' with prob = 0.00

Top 5 Labels for image 'End of all speed and passing limits':
 - 'End of all speed and passing limits' with prob = 1.00
 - 'Priority road' with prob = 0.00
 - 'End of no passing' with prob = 0.00
 - 'End of no passing by vehicles over 3.5 metric tons' with prob = 0.00
 - 'Turn right ahead' with prob = 0.00

Top 5 Labels for image 'Speed limit (50km/h)':
 - 'Speed limit (50km/h)' with prob = 0.92
 - 'Speed limit (30km/h)' with prob = 0.08
 - 'Speed limit (20km/h)' with prob = 0.00
 - 'Speed limit (70km/h)' with prob = 0.00
 - 'Stop' with prob = 0.00

Top 5 Labels for image 'Go straight or left':
 - 'Keep right' with prob = 0.79
 - 'Turn left ahead' with prob = 0.21
 - 'Speed limit (50km/h)' with prob = 0.00
 - 'Speed limit (30km/h)' with prob = 0.00
 - 'Ahead only' with prob = 0.00

Top 5 Labels for image 'Speed limit (30km/h)':
 - 'Speed limit (30km/h)' with prob = 1.00
 - 'Speed limit (80km/h)' with prob = 0.00
 - 'Speed limit (50km/h)' with prob = 0.00
 - 'End of speed limit (80km/h)' with prob = 0.00
 - 'Speed limit (60km/h)' with prob = 0.00

Top 5 Labels for image 'Yield':
 - 'Yield' with prob = 1.00
 - 'Priority road' with prob = 0.00
 - 'End of no passing by vehicles over 3.5 metric tons' with prob = 0.00
 - 'Right-of-way at the next intersection' with prob = 0.00
 - 'Road work' with prob = 0.00

Top 5 Labels for image 'Speed limit (80km/h)':
 - 'Yield' with prob = 1.00
 - 'Keep right' with prob = 0.00
 - 'Stop' with prob = 0.00
 - 'Keep left' with prob = 0.00
 - 'No passing for vehicles over 3.5 metric tons' with prob = 0.00

Top 5 Labels for image 'Speed limit (50km/h)':
 - 'Speed limit (50km/h)' with prob = 1.00
 - 'Speed limit (30km/h)' with prob = 0.00
 - 'Speed limit (80km/h)' with prob = 0.00
 - 'Speed limit (60km/h)' with prob = 0.00
 - 'Keep right' with prob = 0.00

Top 5 Labels for image 'Children crossing':
 - 'Speed limit (30km/h)' with prob = 0.55
 - 'End of speed limit (80km/h)' with prob = 0.38
 - 'Right-of-way at the next intersection' with prob = 0.05
 - 'Speed limit (80km/h)' with prob = 0.01
 - 'Speed limit (60km/h)' with prob = 0.01

Top 5 Labels for image 'No entry':
 - 'No entry' with prob = 1.00
 - 'Keep right' with prob = 0.00
 - 'Stop' with prob = 0.00
 - 'Yield' with prob = 0.00
 - 'End of no passing' with prob = 0.00

The model was able to correctly guess 9 of the 14 traffic signs, which gives an accuracy of ~64%.
