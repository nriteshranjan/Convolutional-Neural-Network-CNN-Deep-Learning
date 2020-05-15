# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:35:03 2020

@author: rrite
"""


# import librarires
import torch
import numpy as np
from IPython import get_ipython
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
from torchvision import datasets
import torchvision.transforms as transforms

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transforms)
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transforms)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)
# Visualising the imageset
import matplotlib.pyplot as plt

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
"""images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize = (25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xsticks = [], ysticks = [])
    ax.imshow(np.squeeze(images[idx]), cmap = 'gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))"""