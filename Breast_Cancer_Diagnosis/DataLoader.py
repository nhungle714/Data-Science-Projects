import pandas as pd
from sklearn.utils import shuffle
import os
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
import pydicom


##### The first method reads and transforms images once then stored the transformed images for use later. 
##### This makes it faster but costs a lot of memory

class MammogramDataset_TL(Dataset):

    def __init__(self, csv_file, root_dir, image_column, num_channel, transform=None,
                transform_type = 'Custom', transform_prob=0.5):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column (string): name of the column image used
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_column = image_column
        self.num_channel = num_channel
        self.transform_prob = transform_prob
        self.transform_type = transform_type
        self.samples = []
        
        for idx in range(len(self.data_frame)):
            image_name = os.path.join(self.root_dir,
                                    self.data_frame.loc[idx, image_column])

            image = pydicom.dcmread(image_name).pixel_array
            
            if self.num_channel > 1:
                image = np.uint8(image/65535*255)
                image = np.repeat(image[...,None],self.num_channel,axis=-1)
            else:
                h,w = image.shape
                resized_h = 1024
                resized_w = int(resized_h/h*w)
                image = transform.resize(image, (resized_h, resized_w), anti_aliasing=True,mode='constant')
                pad_col = resized_h-resized_w
                image = np.pad(image,((0,0),(0,pad_col)),mode='constant',constant_values=0)
                image = (image - image.mean()) / image.std()
                image = image[None,...]

            image_class = self.data_frame.loc[idx, 'class']

            if self.transform:
                image = self.transform(image)
            elif self.transform_type == 'Custom':
                p1 = random.uniform(0, 1)
                p2 = random.uniform(0, 1)
                if p1 <= self.transform_prob:
                    image = image[:,:,-1].copy()
                if p2 <= self.transform_prob:
                    image = transform.rotate(image,180)

            sample = {'x': image, 'y': image_class}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.samples[idx]

def GetDataLoader_TL(train_csv, validation_csv, test_csv, 
                     root_dir, image_column, num_channel, 
                     transform_type, transform_prob, 
               train_transform, validation_transform, 
               batch_size, shuffle, num_workers): 

    train_data = MammogramDataset_TL(csv_file = train_csv, 
                              root_dir = root_image,
                              image_column = image_column,
                              num_channel = num_channel, 
                               transform=train_transform, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    val_data = MammogramDataset_TL(csv_file = validation_csv, 
                            root_dir = root_image,
                            image_column = image_column,
                            transform = validation_transform,
                                 num_channel = num_channel, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    test_data = MammogramDataset_TL(csv_file = test_csv, 
                            root_dir = root_image,
                            image_column = image_column,
                            transform = validation_transform,
                            num_channel = num_channel, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    
    image_datasets = {'train': train_data, 'val': val_data, 'test': test_data}
#     train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
#                             shuffle = shuffle, num_workers = NUM_WORKERS)
#     val_loader = DataLoader(val_data, batch_size=BATCH_SIZE,
#                             shuffle = shuffle, num_workers = NUM_WORKERS)
#     test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
#                             shuffle = shuffle, num_workers = NUM_WORKERS)
    

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True, 
                                              num_workers=NUM_WORKERS) 
                    for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
#     print(len(image_datasets['train']), 
#           len(image_datasets['val']),
#          len(image_datasets['test']))
    return dataloaders, dataset_sizes



### When we do not have enough memory, apply the following approach to get data

# ### The less memory method
# class MammogramDataset_TL(Dataset):   
#     def __init__(self, csv_file, root_dir, image_column, num_channel=1, transform = None, transform_type = 'Custom', transform_prob=0.5):
#         """
#         Args:
#             csv_file (string): Path to the csv file filename information.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#             image_column: column name from csv file where we take the file path
#         """
#         #self.data_frame = pickle.load(open(os.path.join(root_dir,data_file),"rb"))
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_column = image_column
#         self.num_channel = num_channel
#         self.transform_prob = transform_prob
#         self.transform_type = transform_type

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, str(self.data_frame.loc[idx, self.image_column]))
#         image = pydicom.dcmread(img_name).pixel_array
#         if self.num_channel > 1:
#             image = np.uint8(image/65535*255)
#             image = np.repeat(image[...,None],self.num_channel,axis=-1)
#         else:
#             h,w = image.shape
#             resized_h = 1024
#             resized_w = int(resized_h/h*w)
#             image = transform.resize(image, (resized_h, resized_w), anti_aliasing=True,mode='constant')
#             pad_col = resized_h-resized_w
#             image = np.pad(image,((0,0),(0,pad_col)),mode='constant',constant_values=0)
#             image = (image - image.mean()) / image.std()
#             image = image[None,...]
        
#         image_class = self.data_frame.loc[idx, 'class']

#         if self.transform:
#             image = self.transform(image)
#         elif self.transform_type == 'Custom':
#             p1 = random.uniform(0, 1)
#             p2 = random.uniform(0, 1)
#             if p1 <= self.transform_prob:
#                 image = image[:,:,-1].copy()
#             if p2 <= self.transform_prob:
#                 image = transform.rotate(image,180)
        
#         sample = {'x': image, 'y': image_class}

#         return sample
