# Developing An Image Classifier
## Deep Learning

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at.


### Dataset
[The dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) includes **102** flower categories, you can see a few examples below.

![sample_img](https://github.com/yanglinjing/dsnd_p2_img_classifier/blob/master/assets/Flowers.png)

It contains 3 folders:
- train: 6552 images,
- valid: 818 images,
- test: 819 images.

### Progress
The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on the dataset
* Use the trained classifier to predict image content

#### Libraries
```
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models

import json
```

#### Parameters
```
batch_size = 32
learning_rate = 0.001
drop_out = 0.5
epochs = 5
```
#### Transfer models
I loaded a pretrained model `torchvision.models.vgg16` to get the image features, then built and trained a new feed-forward classifier using those features.

#### Implementation
I wrote an application that consists of Python scripts (in the 'py' folder) which could be run from the command line.

It helps classify flower pictures, and the outcome is as follows:

![output](https://github.com/yanglinjing/dsnd_p2_img_classifier/blob/master/assets/output.png)
