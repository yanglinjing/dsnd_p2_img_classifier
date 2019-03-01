%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json
import arg_parse
import utils_predict
from utils_train import build_model

pretrained_mean = [0.485, 0.456, 0.406]
pretrained_std = [0.229, 0.224, 0.225]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# to GPU
cpu = torch.device("cpu" if device.type == 'cuda' else 'cpu')# to CPU

# parameters
batch_size = args.batch_size
learning_rate = args.learning_rate
drop_out = args.drop_out
epochs = args.epochs
print_every = args.print_every


# model settings
arch = args.arch
input_size = args.input_size
output_size = args.output_size
hidden_layers = args.hidden_layers


# paths
checkpoint_path = args.checkpoint_path
data_dir = args.data_dir
image_path = args.image_path

# others
topk = args.topk


def main(checkpoint_path,
         image_path):

  # rebuild model by using checkpoint.pth
  model, optimizer, criterion = u.rebuild_model(checkpoint_path)

  img = process_image(image_path, pretrained_mean, pretrained_std)

  # predict probabilities & classes
  prediction = predict(img, model, device, topk)

  # a dict like {'15': 'yellow iris'}
  with open('cat_to_name.json', 'r') as f:
    mapper = json.load(f)

  # plot
  plot_classification(image_path, 
                      prediction,
                      mapper,
                      pretrained_mean,
                      pretrained_std)


if __name__== "__main__":
    main()
