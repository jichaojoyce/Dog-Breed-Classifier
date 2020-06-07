# Data Scientist Nanodegree
## Capstone Project
Chao Ji 
June 7, 2020

![Definition Pic](/doc/sample_dog_output.png)

## I. Definition
### Project Overview
When you are walking, a cuty dog comes and you are curious about its breed. Do you have an experience like that? I have definetly... To solve the real-world images classifier problem, this project uses Convolutional Neural Networks (CNNs) and transfer learning to build a pipeline to process real-world, user-supplied images. This project aims to create a web application that is able to identify a breed of dog if given a photo or image as input.If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 
### Problem Statement
The goal is to create a web application to (1) recognize whether the picture is a human or a dog; and (2)identify a breed of dog or a resembled dog breed of human. This is a multi-class classification problem. Differenting between breeds is a difficult problem. That is, the breeds are not the obvious characteristics that can be observed as the size, shape and color. Consider that even a human would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel. It is not difficult to find other dog breed pairs with minimal inter-class variation. Also, even the same breed, the dogs' color may be different. But this project is meaningful that same methods can be applied to identify breeds of other species like plants and birds. To solve the problem, CNNs and transfer learning are applied because their advantages to assist with keypoint detection in dogs, namely in indentifying eyes, nose, and ears. 

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
* CNN architecture 

The images are dividied every pixel in every image by 255. A CNN achitecture is established. The network I chose has 3 convolution layers and 3 maximum pooling layers to reduce the dimensionality and increase the depth. The filters used were 16, 32, 64 respectively.

The  final layer has 133 nodes to match our classes of dog breeds and a softmax activation loss function was obtained to estimate probabilities for each of the classes.

![layer Pic](/doc/layers.png)

The target was to to achieve a CNN with >1% accuracy. The network described above achieved 1.0766% without any fine-tuning of parameters and without any augmentation on the data. This limited accuracy may due to I only used 10 epochs and the limited layer.

* Train a CNN using transfer learning
Several pre-trained networks models such as VGG-16, VGG-19, and ResNet-50 are used as a fixed feature extractor for use with keras. The last convolutional output of the pre-trained models is fed as input to our model as the first layer for extra training. A global average pooling layer and a fully connected layer are added to the model structure. 

![layer Pic](/doc/layers1.png)

The test accuracy using VGG-16 is 39.71%. Similar approaches are applied to add the VGG-19 and ResNet-50 models as the first layer in below figures. The test accuracy using VGG-19 and ResNet-50 are 49.16% and 81.22% respectively. The transfer training does improve the accuracy and speed. Hence, the ResNet-50 model is applied for the next dog breed classification prediction. 

![layer Pic](/doc/layers2.png)

![layer Pic](/doc/layers3.png)

* Overall classification

The new model using the ResNet-50 model as the first layer is well trained. Then algorithms are wrote to help classify the pictures. The concepts are: 
1. first determines whether the image contains a human, dog, or neither using the pre-defined 'dog_detector' and 'face_detector'.
2. if human or dog is defined, then the pre-trained model is used to predict the similar dog-breed for human and the dog-breed for dog. 

## IV. Results
### Model Evaluation and Validation
Several figures are tested. 
![layer Pic](/results/result1.png)
![layer Pic](/results/result2.png)
![layer Pic](/results/result3.png)
![layer Pic](/results/result4.png)
![layer Pic](/results/result5.png)
![layer Pic](/results/result6.png)
### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
