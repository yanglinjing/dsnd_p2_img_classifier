import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json
from utils_train import build_model


def rebuild_model(checkpoint_path):
  # load checkpoint
  checkpoint = torch.load(checkpoint_path)

  # rebuild model
  model, optimizer, criterion = build_model(checkpoint['arch'],
                                            checkpoint['input_size'],
                                            checkpoint['output_size'],
                                            checkpoint['hidden_layers'],
                                            checkpoint['learning_rate'],
                                            checkpoint['drop_out'])

  model.load_state_dict(checkpoint['state_dict'])
  model.class_to_idx = checkpoint['class_to_idx']

  optimizer.state_dict = checkpoint['optimizer state']

  return model, optimizer, criterion


def process_image(image_path, pretrained_mean, pretrained_std):
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
    mean = np.array(pretrained_mean)
    std = np.array(pretrained_std)

    # subtract means from each color channel, then divide by std
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std


    # reorder dimensions
    np_image = np.transpose(np_image, (2, 0, 1))

    return torch.tensor(np_image)

def predict(img, model, device, topk):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    img = process_image(...)
    '''

    img = img.unsqueeze_(0) # this is for VGG
    img = img.float()

    with torch.no_grad():
        # forward
        output = model(img.to(device))

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)


def imshow(image, ax=None, title=None,
           pretrained_mean,
           pretrained_std):

    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array(pretrained_mean)
    std = np.array(pretrained_std)
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def plot_classification(image_path, prediction, mapper,
                       pretrained_mean, pretrained_std):
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
    ax1.imshow(img, pretrained_mean, pretrained_std)
    ax1.axis('off')

    y_pos = np.arange(len(p))
    ax2.barh(y_pos, p * 100)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in c])
    ax2.invert_yaxis()

    ax2.tick_params(labelsize=14)
    ax2.set_xlabel('Percentage (%)', fontsize = 14)
    ax2.set_title("Perdicted Classes", size = 16)
