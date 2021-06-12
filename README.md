# Artificial Intelligence Project Driver Drowsiness Detection
The objective of this project is to build a drowsiness detection system that will detect that a person’s eyes are closed for a few seconds. This system will alert the driver when drowsiness is detected and can prevent accidents.
The majority of accidents happen due to the drowsiness of the driver. So, to prevent these accidents we will build a system using Python, OpenCV, and Keras which will alert the driver when he feels sleepy.
##  Requirements
### Hardware Requirements:
* Processor: Quad-core Intel Core i7 or higher.
* Ram: 8Gb DDR4 or higher.
* SSD: 256 Gb or higher.
* Graphic Card: GTX 980 or higher.
* Camera Resolution: 4mp or higher
### Software Requirements:
* Operating System: 
* Windows 8 or higher
* Ubuntu 16.04 or higher
* MacOS X 10.6 or higher
* Python Interpreter:
* Python Version: 3.7 or higher
## Working
### Dataset generation
We selected the dataset consisting of the images of eyes with different properties. The resource for this data set is given as: http://mrl.cs.vsb.cz/eyedataset
The dataset that we selected consists of 84898 images of human eyes obtained through an eye detector that used a histogram of oriented gradients combined with a support vector machine classifier. The image subjects were 33 men and 4 women. There are also the images with glasses on, reflection of light (which can be none, small or big). The all images are monochrome images and there are 3 different types of sensors which are Realsense, IDS and Aptina from which images are created. An example image of the dataset is given as below

![image](https://user-images.githubusercontent.com/37701187/121773130-43058b80-cb93-11eb-92e2-424bc11de126.png)
### Model Architecture
Our model is based on the CNN it has total of 128 nodes and the distribution of layers are described as below:
There are 32 nodes in one convolutional layer.
32 nodes in another convolutional layer.
In the last convolutional layer there are 64 nodes.
Then a fully connected layer of 128 nodes.
There is another layer which is actually an output layer and a softmax function is applied on it, while relu function is applied on the above mentioned 3 convolutional layers. Kernel size in the convolutional layers is 3.

### Driver Implementation:
#### Opening Video Camera and Fame Conversion:
To take input images first the program grabs a video capture using OpenCV built-in function VideoCapture. It returns a video capture of as a capture object which has a method of capread which reads the frame and then image is stored manually.
#### Object Segmentation and Extraction of Region of Interest:
To segment the faces we have used a cascade classifier to which we feed a haar cascade file for the face. The steps are mentioned below
First Convert Binary Images to Monochrome Images(Grayscale)
Then make a face classifier using a cascade classifier.
Then we used detectmultiscale to get the axis points of the region of interest i.e. face.
To extract the eyes only we did the above steps but with a haar cascade file of left eye and right eye.
#### Eye State Classification:
To classify the state of eyes we used our model which was trained for eye classification. To classify the eye state we have done following steps:
Convert RGB image to Monochrome image.
Resize the image dimensions to 24 by 24 as our model trained on the 24 by 24 images.
Import the model by load_model function.
Then predict eyes one by one by using the function model.predict_classes

#### Score Computation
After classifying we need to maintain some sort of book keeping in the form of score so that when the score reaches a certain threshold it shoots off the alarm. To compute the score we maintain a count if more than 15 seconds a user has his eyes closed, that is in almost all the frames of 15 seconds the person’s eyes are closed then the alarm will start, else it will not ring.

## Results
During the development phase the project has gone through various experimentations and the results of models as well as driver function is mentioned below.
5.1 Model Accuracy:
Model is trained in 15 iterations and after 15 iterations the result accuracy is given in the picture below:
![ads](https://user-images.githubusercontent.com/37701187/121773266-3170b380-cb94-11eb-8dfa-26df0cd9b4af.png)

## Overall Poster of the Project
![image](https://user-images.githubusercontent.com/37701187/121773285-6bda5080-cb94-11eb-9682-80a6081ff6b2.png)



