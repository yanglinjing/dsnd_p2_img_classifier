# Developing An Image Classifier
### (Deep Learning)

# Introduction

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at.


## Data
[The dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) includes **102** flower categories, you can see a few examples below.

![sample_img](https://github.com/yanglinjing/dsnd_p2_img_classifier/blob/master/assets/Flowers.png)

It contains 3 folders:
- train: 6552 images,
- valid: 818 images,
- test: 819 images.

## Progress
The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on the dataset
* Use the trained classifier to predict image content


## Installation
Language: Python 3

Software: Google Colab | Jupyter Notebook

### Libraries
```
import numpy as np
import time
import arg_parse
import json
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

```

### Parameters
```
batch_size = 32
learning_rate = 0.001
drop_out = 0.5
epochs = 3
```

### GPU & CPU

GPU has been used here.
```
# to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# to CPU
cpu = torch.device("cpu" if device.type == 'cuda' else 'cpu')
```


## Supportive Documents
`cat_to_name.json`: since image categories are presented by numbers (e.g. *1, 5, 20*), this `.json` file shows both the number and name of the categories (e.g. "*21: fire lily*")

```
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)   
```

## Implementation
I wrote an application that consists of Python scripts (in the 'py' folder) which could be run from the command line.

It helps classify flower pictures, and the outcome is as follows:

![output](https://github.com/yanglinjing/dsnd_p2_img_classifier/blob/master/assets/output.png)

In `py` folder, there are 2 important files `train.py` and `predict.py`.
- `train.py` will train a new network on a dataset and save the model as a checkpoint.
- `predict.py` uses a trained network to predict the class for an input image.
- Other  `.py` files in this folder are supportive documents for them.



# Process
I loaded a pretrained model `torchvision.models.vgg16`  to get the image features, then built and trained a new feed-forward classifier using those features (Find more about [torchvision](http://pytorch.org/docs/0.3.0/torchvision/index.html)).

(Note: Only some important codes are included here. For all the cods, please see the `.ipynb` document.)

## Step 1. Load & Transform Data

The **pre-trained** networks were trained on the ImageNet dataset where each color channel was ***normalized*** separately. For all three sets, the means and standard deviations of the images are normalized to what the network expects. These values will shift each color channel to be centered at 0 and range ***from -1 to 1***.

```
pretrained_mean = [0.485, 0.456, 0.406]
pretrained_std = [0.229, 0.224, 0.225]
```
### Step1.1 Transform Data
The dataset has already been split into three parts, **training**, **validation**, and **testing**.

- For the **training**, I applied transformations such as ***random scaling, cropping, and flipping***, which helped the network generalise leading to better performance. I also need to  ***resized*** the input data is to **224x224** pixels as required by the pre-trained networks.

- For the **validation** and **testing** sets, I just  ***resized*** then ***cropped*** the images,  as they were used to measure the model's performance.

```
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_mean, std=pretrained_std)
     ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_mean, std=pretrained_std)
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_mean, std=pretrained_std)
     ])
}
```


### Step 1.2 Load Data

-  Load the datasets with `ImageFolder`

```
data_dir = 'flowers'

image_datasets = {
    x: datasets.ImageFolder(root = data_dir + '/' + x,
                            transform = data_transforms[x])
    for x in list(data_transforms.keys())
}
```

- Define the dataloaders
```
dataloaders = {
    x : torch.utils.data.DataLoader(image_datasets[x],
                                    batch_size = batch_size,
                                    shuffle = True)
    for x in list(data_transforms.keys())
}
```

## Step 2.  Building & Training the Classifier

Now that the data is ready, I used one of the pretrained models from `torchvision.models` to get the image **features**, then built and trained a new feed-forward classifier using those features.

The steps are as follows:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best ***hyperparameters***


### Step 2.1.  Load a pre-trained network - VGG

```
def load_model(arch='vgg16'):

    # load the model trained on ImageNet: VGG or alexnet

    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('{} is not available. Please choose "vgg19" or "alexnet"'.format(arch))

    return model
```

- Now we can check the original classifiers of the model `vgg16`:
```
load_model().classifier
```
OUTPUT:

> (0): Linear(in_features=25088, out_features=4096, bias=True)

> (1): ReLU(inplace)

> (2): Dropout(p=0.5)

> (3): Linear(in_features=4096, out_features=4096, bias=True)

> (4): ReLU(inplace)

> (5): Dropout(p=0.5)

> (6): Linear(in_features=4096, out_features=1000, bias=True)


- We need to keep the input size same as the original one. Since we have 102 flower categories, our output size should be 102.

```
arch = 'vgg16'
input_size = 25088
hidden_layers = 4096
output_size = 102
```


### Step 2.2.  Define a New Model
- Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout

```
def build_model(arch,
                input_size,
                output_size,
                hidden_layers,
                learning_rate,
                drop_out):

  # load the model trained on ImageNet
  model = load_model(arch)

  # Freezing Parameters
  for param in model.parameters(): # instead of using for param in model.features
    param.require_grad = False # Turn off gradients for the model

  # Define new classifier
  classifier = nn.Sequential(nn.Linear(input_size, hidden_layers),
                             nn.ReLU(),
                             nn.Dropout(p = drop_out),
                             nn.Linear(hidden_layers, output_size),
                             nn.LogSoftmax(dim = 1))


  # update model
  model.classifier = classifier

  # put model into GPU
  model.to(device)

  # Define Loss
  criterion = nn.NLLLoss()

  # Define Optimiser
  optimizer = optim.Adam(model.classifier.parameters(), # use parameters in model
                         lr = learning_rate) # learning rate

  return model, optimizer, criterion
 ```


 ### Step 2.3.  Train &  Validation
- **Train** the classifier layers using backpropagation using the pre-trained network to get the features
- Track the loss and accuracy on the **validation** set to determine the best hyperparameters

```
def validation(phase,
               dataloaders,
               device,
               model,
               criterion):

  test_loss = 0
  accuracy = 0

  # turn off dropout
  model.eval()

  with torch.no_grad():

    # use valid data
    for images, labels in dataloaders[phase]:

      # transfer tensors to GPU
      images, labels = images.to(device), labels.to(device)

      #1 Calculate Loss
      logps = model(images)
      test_loss += criterion(logps, labels).item()

      #2 Calculate Accuracy
      ps = torch.exp(logps)
      # Our model returns logSoftmax, which is Log probability of the classes, so we need exp of it

      ## get 1st largest value & probability  
      ## check equality with labels: whether predicted classes match actual ones
      equality = (labels.data == ps.max(1)[1]) # dim=1 column

      ## calculate accuracy by equality
      accuracy += equality.type_as(torch.FloatTensor()).mean()

  return test_loss,  accuracy

```

```
def train(dataloaders,
         device,
         model,
         criterion,
         optimizer,
         epochs,
         print_every = 40): # print records after every 40 images

  start = time.time()

  # Define variables to be used in training
  steps = 0
  running_loss = 0

  for epoch in range(epochs):

    # use training data
    for images, labels in dataloaders['train']:

      # transfer tensors to GPU
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad() # gradient = 0

      steps += 1

      #1 forward
      logps = model(images) # log probabilities

      #2 calculate loss
      loss = criterion(logps, labels)

      #3 backward
      loss.backward()

      #4 Take an update step and few the new weights
      optimizer.step()

      # keep tracking the loss while updating data
      running_loss += loss.item()


      # -----Valid-----
      if steps % print_every == 0:

        test_loss, accuracy = validation('valid',
                                         dataloaders,
                                         device,
                                         model,
                                         criterion)

        # print average loss /accuracy, time cost
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train loss: {running_loss/print_every:.3f} | "
              f"Valid loss: {test_loss/len(dataloaders['valid']):.3f} | "
              f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")

        # prepare to restart training
        running_loss = 0
        model.train()

  time_total = time.time() - start
  print(f'\nTraining has been completed in {time_total//60:.0f}m {time_total%60:.0f}s.')

```

OUTPUT:

> Epoch 3/3 | Train loss: 1.259 | Valid loss: 0.537 | Valid accuracy: 0.849

> Training complete in 48m 9s


## Step 3. Testing

Run the test images through the trained network and measure the accuracy, in the same way as how we did in the validation step.

```
test_loss, accuracy = validation('test',
                                 dataloaders,
                                 device,
                                 model,
                                 criterion)

```
OUTPUT:
> Test loss: 0.614   |   Test accuracy: 0.832


## Step 4. Save the checkpoint

Save the model (both model architecture and parameters) to a `.pth` file for making predictions later.
```
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict(),
    'optimizer state': optimizer.state_dict,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'drop_out': drop_out,
    'epochs': epochs,
    'arch': 'vgg16',
    'input_size': input_size,
    'output_size': output_size,
    'hidden_layers': hidden_layers,
    'pretrained_mean': pretrained_mean,
    'pretrained_std': pretrained_std
}

torch.save(checkpoint, checkpoint_path)
```


## Step 5.  Inference for classification
In this step, a `predict` function has been written - when a flower image has been passed into it, it can the top $K$ most likely classes  along with their probabilities.

### Step 5.1. Image Preprocessing

Write a function to  **process the images** in the **same** manner used for **training**.

1. Use [`PIL`](https://pillow.readthedocs.io/en/latest/reference/Image.html) to load the image

2. [`Resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) the images where the shortest side is 256 pixels, keeping the aspect ratio; then crop out the center 224x224 portion of the image.

3. Convert the color channels from integers to floats, and then put them into a Numpy array.

4. Normalise color channels from 0-255 to 0-1.

5. Reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html), since PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array.

```
def process_image(image_path, mean = pretrained_mean, std = pretrained_std):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = Image.open(image_path)

    # Resize as the same manner of training
    img_loader = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])



    #Color channels are integers 0-255, but the model expected floats 0-1
    pil_image = img_loader(pil_image).float()

    # change to Np
    np_image = np.array(pil_image)    

    # Normalise
    mean = np.array(mean)
    std = np.array(std)

    # subtract means from each color channel, then divide by std
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std   


    # reorder dimensions
    np_image = np.transpose(np_image, (2, 0, 1))

    return torch.tensor(np_image)
```


## Class Prediction

Write a function for making predictions, which takes a path to an image and a model checkpoint, then return the probabilities and classes.


- Get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk), which returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes.

- Convert from these indices to the actual class labels using `class_to_idx` which has been added to the model.

```
def predict(image_path, model, device, topk=5):

    img = process_image(image_path)
    img = img.unsqueeze_(0) # this is for VGG
    img = img.float()

    with torch.no_grad():
        # forward
        output = model(img.to(device))

    probability = F.softmax(output.data, dim=1)

    return probability.topk(topk)
 ```



 ## Sanity Checking
Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image.

```
def plot_classification(image_path, prediction, mapper):
    '''
    To view an image & its predicted classes.

    + prediction- results of function predict(), including probabilities & classes
    + mapper - a dictionary like {'15': 'yellow iris'} from 'label mapping' section.
    '''


    p = np.array(prediction[0][0].to(cpu)) #probabilities
    c = np.array(prediction[1][0].to(cpu)) #classes
    c = [str(x) for x in c] # convert classes from number to str


    img = Image.open(image_path)

    flower_class = image_path.split('/')[-2] # get the class number
    flower_name = mapper[flower_class] # put the class number into name


    fig, (ax1, ax2) = plt.subplots(figsize=(18,6),
                                   ncols=2, nrows=1)

    ax1.set_title(flower_name, size = 16)
    ax1.imshow(img)
    ax1.axis('off')

    y_pos = np.arange(len(p))
    ax2.barh(y_pos, p * 100)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in c])
    ax2.invert_yaxis()

    ax2.tick_params(labelsize=14)
    ax2.set_xlabel('Percentage (%)', fontsize = 14)
    ax2.set_title("Perdicted Classes", size = 16)
```
