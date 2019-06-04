
This project focuses on classifiying the lung disease using chest x-ray dataset provided by NIH (https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community). The goal is to classify pneumothorax and cardiomegaly cases, using deep CNN models 1) customized convolutional ResNet and 2) transfered learning resnet 34. I used HPC at NYU to train the models. 

Architecture of the Customized convolutional ResNet model: 
  - 2 convolutional layers
  - 2 residual units (similar to Figure 2 of https://arxiv.org/pdf/1512.03385.pdf)
  - 1 fully connected layer 
  - 1 classification layer
      -  3x3 convolution kernels (stride 1 in resnet units and stride 2 in convolutional layers)
      -  ReLU for an activation function
      -  max pooling with kernel 2x2 and stride 2 only after the convolutional layers. 
      -  The number of feature maps in hidden layers as: 16, 16, 16, 32, 32, 32, 64 (1st layer, ..., 7th layer). 

Input --> Convolution1 --> ResNetBlock1 --> Convolution2 --> ResNetBlock2 --> FC --> Output

This folder consists of python scripts for functions used, and complete python scripts to train ConvNet model and transfer-learning ResNet34 model on HPC. 

Reference: Wang X, Peng T, Lu L, et al. 
ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR. 2017. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8099852
