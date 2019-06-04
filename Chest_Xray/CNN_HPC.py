from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from skimage import io
import torch
from torchvision import transforms
import torchvision
from skimage import color
import copy

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage import io, transform
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import scipy
import random
import pickle
import scipy.io as sio
import itertools
from scipy.ndimage.interpolation import shift
import copy
import warnings
#warnings.filterwarnings("ignore")
plt.ion()
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class Conv_ResNet(nn.Module):
    def __init__(self):
        super(Conv_ResNet, self).__init__()
        
        self.conv1 = nn.Sequential(         # image shape (1, 1024, 1024)
            nn.Conv2d(
                in_channels=1,              # number of input channels
                out_channels=16,            # number of output filters
                kernel_size=3,              
                stride=2                # stride ofor the conv operation                 
            ),                              # output shape (16, 1024, 1024)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 255, 255)
        )
        
        
        self.res1_conv1 = nn.Conv2d(16, 16, 3, stride= 1, dilation = 1, padding= 1)
        self.res1_bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Sequential(         # image shape (16, 255, 255)
            nn.Conv2d(
                in_channels=16,              # number of input channels
                out_channels=32,            # number of output filters
                kernel_size=3,              
                stride=2                # stride ofor the conv operation                 
            ),                              # 
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 63, 63)
        )
        
        
        self.res2_conv1 = nn.Conv2d(32, 32, 3, stride= 1, dilation = 1, padding= 1)
        self.res2_bn1 = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(127008, 64)
        self.out = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        
    def forward(self,x): 
        x = self.conv1(x)
        
        residual = x
        image_res1 = self.res1_conv1(x)
        image_res1 = self.res1_bn1(image_res1)
        image_res1 = self.relu(image_res1)
        image_res1 = self.res1_conv1(image_res1)
        image_res1 = self.res1_bn1(image_res1)
        image_res1 += residual
        image_res1 = self.relu(image_res1)
        
        image_res1 = self.conv2(image_res1)
        
        residual = image_res1
        image_res2 = self.res2_conv1(image_res1)
        image_res2 = self.res2_bn1(image_res2)
        image_res2 = self.relu(image_res2)
        image_res2 = self.res2_conv1(image_res2)
        image_res2 = self.res2_bn1(image_res2)
        image_res2 += residual
        image_res2 = self.relu(image_res2)
        
        image_res2 = image_res2.view(image_res2.size(0), -1)
        image_fc = self.fc1(image_res2)
        
        final = self.relu(image_fc)
        final = self.out(final)
        
        return final

class Convnet5Layer(nn.Module,):
    def __init__(self,fc_size=32):
        super(Convnet5Layer, self).__init__()

        self.conv1 = nn.Sequential(         # image shape (1, 1024, 1024)
            nn.Conv2d(
                in_channels=1,              # number of input channels
                out_channels=16,            # number of output filters
                kernel_size=3,              # filter size yo ucan also specify it as (3,3)
                stride=1,                   # stride ofor the conv operation
                padding=1,                  # if want same width and length of this image after con2d
            ),                              # output shape (16, 1024, 1024)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 512, 512)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # output shape (16, 256, 256)
        )
        
        self.res1_conv1 = nn.Conv2d(16, 16, 3, stride= 1, dilation = 1, padding= 1)
        self.res1_bn1 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # output shape (32, 128, 128)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # output shape (32, 64, 64)
        )
        
        self.res2_conv1 = nn.Conv2d(32, 32, 3, stride= 1, dilation = 1, padding= 1)
        self.res2_bn1 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # output shape (64, 32, 32)
        )
        
        self.fc = nn.Linear(64 * 32 * 32 , fc_size)
        
        self.out = nn.Linear(fc_size , 2)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        residual = x
        image_res1 = self.res1_conv1(x)
        image_res1 = self.res1_bn1(image_res1)
        image_res1 = self.relu(image_res1)
        image_res1 = self.res1_conv1(image_res1)
        image_res1 = self.res1_bn1(image_res1)
        image_res1 += residual
        image_res1 = self.relu(image_res1)

        image_res1 = self.conv3(image_res1)
        image_res1 = self.conv4(image_res1)

        residual = image_res1
        image_res2 = self.res2_conv1(image_res1)
        image_res2 = self.res2_bn1(image_res2)
        image_res2 = self.relu(image_res2)
        image_res2 = self.res2_conv1(image_res2)
        image_res2 = self.res2_bn1(image_res2)
        image_res2 += residual
        image_res2 = self.relu(image_res2)

        image_res2 = self.conv5(image_res2)

        image_res2 = image_res2.view(image_res2.size(0), -1)
        image_fc = self.fc(image_res2)
        
        final = self.relu(image_fc)
        final = self.out(final)


        return final