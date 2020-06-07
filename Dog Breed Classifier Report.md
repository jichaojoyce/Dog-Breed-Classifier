# Data Scientist Nanodegree
## Capstone Project: Dog Breed Classifier 
Chao Ji 
June 7, 2020

![Definition Pic](/doc/sample_dog_output.png)

## I. Definition
### Project Overview
When you are walking, a cute dog comes and you are curious about its breed. Do you have an experience like that? I have definitely... To solve the real-world images classifier problem, this project uses Convolutional Neural Networks (CNNs) and transfer learning to build a pipeline to process real-world, user-supplied images. This project aims to create a web application that is able to identify a breed of dog if given a photo or image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 
### Problem Statement
The goal is to create a web application to (1) recognize whether the picture is a human or a dog; and (2)identify a breed of dog or a resembled dog breed of human. This is a multi-class classification problem. Differentiating between breeds is a difficult problem. That is, the breeds are not the obvious characteristics that can be observed as the size, shape, and color. Consider that even a human would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel. It is not difficult to find other dog breed pairs with a minimal inter-class variation. Also, even the same breed, the dogs' color may be different. But this project is meaningful that the same methods can be applied to identify breeds of other species like plants and birds. To solve the problem, CNNs and transfer learning are applied because of their advantages to assist with keypoint detection in dogs, namely in identifying eyes, nose, and ears. 

The tasks involved are the following:

Step 0: Import Datasets

Step 1: Detect Humans

Step 2: Detect Dogs

Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)

Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

Step 6: Write your Algorithm

Step 7: Test Your Algorithm

### Metrics
* Accuracy

Accuracy is a common metric for classifiers, defined as the true breed prediction among the whole predictions.

![\large accuracy = \frac{\text{true breed prediction}}{\text{dataset size}}](https://render.githubusercontent.com/render/math?math=%5Clarge%20accuracy%20%3D%20%5Cfrac%7B%5Ctext%7Btrue%20breed%20prediction%7D%7D%7B%5Ctext%7Bdataset%20size%7D%7D)

Ideally, we would like to create a CNN that can achieve results of over 60% accuary. That is, it can correclt identify the dog breed 6 times out of 10. We will be using the accuracy metric on the testing dataset to measure the model performance. 

* Categorical cross-entroy loss

The categorical cross-entroy loss, also called softmax loss is used to train a CNN to output a probability over the C classes for each image. 
 
![category Pic](/doc/categorical.png)

The RMSprop optimizer is used to mimimum the loss function. Different as the gradient descent, the RMSprop optimizer restricts the oscillations in the vertical direction. 

## II. Analysis
### Data Exploration
The dog pictures and human face for this model cross-validation are downloaded from the below links.

[dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

[human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).

Totally, there are 8351 total dog images, belong to 133 dog categories. In contrast, there are 13233 total human images. The dog dataset is divided into traning, validation and test dataset. Among them, 6680 dog images are for training, 835 dog images are for validation and 836 dog images are for images test. 

### Exploratory Visualization
* Detect Humans
The OpenCV's implementation of Haar feature-based cascade classifiers is used to detect human faces in images. The pre-trained face detector 'haarcascade_frontalface_alt.xml' is used to detect the humans' faces. 

Here 'Number of faces detected: 1'

![human Pic](/doc/humandetector.png)

When using the human files and dog files to test the performace of human face detector. It shows 100.0% accuracy to detect human face in human files and 11.0% detect human face in dog files.

* Detect Dogs
A pre-trained ResNet-50 model is used to detect dogs in images. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image. Similary, the same data in detect humans are used to test the ability of detecting dogs. It shows 0.0% dog face is detected in human_files, and 100.0% dog face is detected in dog_files.

### Algorithms and Techniques
#### Data Preprocessing
* CNN architecture 

The images are dividied every pixel in every image by 255. A CNN achitecture is established. The network I chose has 3 convolution layers and 3 maximum pooling layers to reduce the dimensionality and increase the depth. The filters used were 16, 32, 64 respectively.

The  final layer has 133 nodes to match our classes of dog breeds and a softmax activation loss function was obtained to estimate probabilities for each of the classes.

![layer Pic](/doc/layers.png)

The target was to to achieve a CNN with >1% accuracy. The network described above achieved 1.0766% without any fine-tuning of parameters and without any augmentation on the data. This limited accuracy may due to I only used 10 epochs and the limited layer.

#### Implementation
* Train a CNN using transfer learning
Several pre-trained networks model VGG-16 is used as a fixed feature extractor for use with keras. The last convolutional output of the pre-trained models is fed as input to our model as the first layer for extra training. A global average pooling layer and a fully connected layer are added to the model structure. 

![layer Pic](/doc/layers1.PNG)

The test accuracy using VGG-16 is 39.71%. To increase the accuracy, other models are tested. 

#### Refinement
VGG-19 and ResNet-50 pre-trained models are applied for tranfer learning. Similar approaches are applied to add the VGG-19 and ResNet-50 models as the first layer in below figures. The test accuracy using VGG-19 and ResNet-50 are 49.16% and 81.22% respectively. The transfer training does improve the accuracy and speed. Hence, the ResNet-50 model is applied for the next dog breed classification prediction. 

![layer Pic](/doc/layers2.PNG)

![layer Pic](/doc/layer3.PNG)

* Overall classification

The new model using the ResNet-50 model as the first layer is well trained. Then algorithms are wrote to help classify the pictures. The concepts are: 
1. first determines whether the image contains a human, dog, or neither using the pre-defined 'dog_detector' and 'face_detector'.
2. if human or dog is defined, then the pre-trained model is used to predict the similar dog-breed for human and the dog-breed for dog. 

## IV. Results
### Model Evaluation and Validation
Several figures are tested. 
* Detect Human correct

![layer Pic](/results/Result1.PNG)

* Detect Human correct

![layer Pic](/results/Result2.PNG)

* Detect Dog correct

![layer Pic](/results/Result3.PNG)

* Detect Dog correct

![layer Pic](/results/Result4.PNG)

* Detect Dog Fair

![layer Pic](/results/Result5.PNG)

* Detect Dog correct

![layer Pic](/results/result6.PNG)

### Justification
All the results are reasonable that they agreed with the facts. Only the teddy figure is identified as Cocker_spaniel. Although te teddy and Cocker_spaniel has some common characters but they are still different. This is because there is no Teddy type in the categories and there is no Teddy training figures. More tests may be needed to draw a more general conculsion. 

## V. Conclusion
Overall, we consider our results to be a success given the high number of breeds in this classification problem. The trained model can effectively predict the correct breed within the 133 different breeds contained in the dataset.

### Reflection
This project starts with the dog and human detector, and ends with a well-trained model to classify the figures into dog or human or neither and predict the accurate breed prediction. The overall model accuracy is 81.22% using the ResNet-50 method. This accuracy could still be improved by adding layers, increasing the training data and epoaches, and applying regularizations.

### Improvement
* The model could be improved by adding its ability to classify pictures with noise. 
* The model could be improved by adding more data for training. 
* The model could be improved by refining the minimum algorithms...
-----------

