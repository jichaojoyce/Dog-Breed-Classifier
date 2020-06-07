# Dog-Breed-Classifier

![Intro Pic](/doc/sample_dog_output.png)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Libraries](#library)
	2. [Installing](#installing)
	3. [Instruction](#executing)
	4. [Data distribution](#material)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Examples](#example)
<a name="descripton"></a>
## Description

This dog breed classifier project is part of Data Science Nanodegree Program by Udacity applying the deep learning algorithms.
The initial dataset contains the dog pictures and human face for model training are downloaded from the below links.

[dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

[human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). .  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

This project aims to create a web application that is able to identify a breed of dog if given a photo or image as input. The Convolutional Neural Networks (CNNs) and transfer learning are applied to build a pipeline to process real-world, user-supplied images.

The Project contains six files:

1. 'dog_app.ipynb'. The Jupyter notebook builds the web application step by step.   
2. 'dog_app.html'. The html version of the 'dog_app.ipynb' Jupyter notebook. 
3. 'Dog Breed Classifier Report.md'. The report on this project walking you through the problem and methods of developing the model.
4. 'results' folder. The 6 figures downloaded from Google as inputs for model testing. 
5. 'doc' folder. The folder contains relevant pictures related to the report.
6. 'saved_models' folder. The folder contains the trained four models in this project. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Libraries
* Machine Learning Libraries: sklearn.datasets, keras, numpy, glob, random, cv2, PIL, extract_bottleneck_features
* Database Libraqries: tqdm
* Web App and Data Visualization: matplotlib.pyplot

You also need to download the 'bottleneck_features'.

[VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
https://github.com/jichaojoyce/Dog-Breed-Classifier.git
```
<a name="Instruction"></a>
### Instructions:
1. In a terminal or command window, navigate to the top-level project directory Recommendations-with-IBM/ (that contains this README) and run one of the following commands:
```ipython notebook dog_app.ipynb```

or

```jupyter notebook dog_app.ipynb```

This will open the iPython Notebook software and project file in your browser.
### Data distribution:

Totally, there are 8351 total dog images, belong to 133 dog categories. In contrast, there are 13233 total human images. The dog dataset is divided into traning, validation and test dataset. Among them, 6680 dog images are for training, 835 dog images are for validation and 836 dog images are for images test. 

<a name="authors"></a>
## Authors

* [Chao Ji](https://github.com/jichaojoyce)

<a name="license"></a>
## License
Feel free to use it!
<a name="acknowledgement"></a>
## Acknowledgements

Credicts give to [Udacity](https://www.udacity.com/).

## Examples
A classification example for human figure. 

![example](/results/Result1.PNG)

A classification example for dog figure with Boykin spaniel breed. 

![example](/results/result6.PNG)
