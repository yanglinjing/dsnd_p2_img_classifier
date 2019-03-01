%matplotlib inline
%config InlineBackend.figure_format = 'retina'

#import numpy as np
import time
import matplotlib.pyplot as plt
#from PIL import Image

import torch
from torch import nn, optim
#import torch.nn.functional as F
from torchvision import datasets, transforms, models

#import json
import arg_parse
import utils_train as t

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

def main():

  # load data
  data_transforms = t.transform_data(pretrained_mean, pretrained_std)
  image_datasets = t.load_data(data_dir, pretrained_mean, pretrained_std)
  dataloaders = t.get_dataloader(batch_size, image_datasets)

  # view data
  t.view_data(dataloaders,
              data_dir,
              pretrained_mean,
              pretrained_std)

  # build model
  model, optimizer, criterion = t.build_model(arch,
                                              input_size,
                                              output_size,
                                              hidden_layers,
                                              learning_rate,
                                              drop_out)

  # Training
  t.train(dataloaders,
         device,
         model,
         criterion,
         optimizer,
         epochs,
         print_every)


  # Test
  t.test(dataloaders,
         device,
         model,
         criterion)

  # Save Checkpoint
  t.save_checkpoint(model,
                    optimizer,
                    checkpoint_path,
                    batch_size,
                    learning_rate,
                    drop_out,
                    epochs,
                    arch,
                    input_size,
                    output_size,
                    hidden_layers,
                    pretrained_mean,
                    pretrained_std)

  print("Training has been completed!")


if __name__== "__main__":
    main()
