#import numpy as np
import time
import matplotlib.pyplot as plt
#from PIL import Image

import torch
from torch import nn, optim
#import torch.nn.functional as F

from torchvision import datasets, transforms, models


def transform_data(pretrained_mean, pretrained_std):

    data_transforms = {
      'train': transforms.Compose([
          transforms.RandomRotation(30),# Randomly Rotate 30 degrees
          transforms.RandomResizedCrop(224),#Randomly Resize Img, then Crop to 224 * 224 px
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

    return data_transforms


def load_data(data_dir, data_transforms):

    image_datasets = {
        x: datasets.ImageFolder(root = data_dir + '/' + x,
                                transform = data_transforms[x])
        for x in list(data_transforms.keys())
    }

    return image_datasets


def get_dataloader(batch_size, image_datasets):

    dataloaders = {
      x : torch.utils.data.DataLoader(image_datasets[x],
                                                batch_size = batch_size,
                                                shuffle = True)
      for x in list(data_transforms.keys())
    }

    return dataloaders


def view_data(dataloaders, data_dir, pretrained_mean, pretrained_std):

    image_datasets = load_data(data_dir, pretrained_mean, pretrained_std)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes

    print ("Dasaset Size: "+ str(dataset_sizes) + "\n")
    n_class = len(class_names)
    print ("Number of classes: "+ str(n_class) + "\n")
    print ("Classes: "+ str(class_names) + "\n")

    print (f'Number of dataloaders: {len(dataloaders)}')

    print ('A sample image')
    images, labels = next(iter(dataloaders["train"]))
    print(f'The size of image: {len(images[0,2])}')
    plt.imshow(images[0,0])




def load_model(arch):
    # load the model trained on ImageNet: VGG or alexnet
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('{} is not available. Please choose "vgg16" or "alexnet"'.format(arch))

    return model



def build_model(arch,
                input_size,
                output_size,
                hidden_layers,
                learning_rate,
                drop_out):

  # load the model trained on ImageNet
  model = load_model(arch)

  # Freezing Parameters
  for param in model.features:
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


def train(dataloaders,
         device,
         model,
         criterion,
         optimizer,
         epochs,
         print_every): # print records after every 30 images

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
  print(f'\nTraining complete in {time_total//60:.0f}m {time_total%60:.0f}s')


def test(dataloaders,
         device,
         model,
         criterion):

  test_loss, accuracy = validation('test',
                                 dataloaders,
                                 device,
                                 model,
                                 criterion)

  print(f"Test loss: {test_loss/len(dataloaders['valid']):.3f}.. "
        f"Test accuracy: {accuracy/len(dataloaders['valid']):.3f}") # average loss /accuray



def save_checkpoint(image_datasets,
                    model,
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
                    pretrained_std):

  # the mapping of classes to indices which you get from one of the image datasets
  model.class_to_idx = image_datasets['train'].class_to_idx

  # To save
  # 1. model architecture,
  # 2. state dict.


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

  # Save to file
  torch.save(checkpoint, checkpoint_path) # .pth - pytorch checkpoints
