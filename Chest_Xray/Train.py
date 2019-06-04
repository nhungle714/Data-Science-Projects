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


def train_model(model, criterion, optimizer, num_epochs, data_sizes,
               trainVal = ['train', 'val'], verbose = True): 
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    best_loss = np.inf
    loss_hist = {'train': [], 'val': []}
    accuracy_hist = {'train': [], 'val': []}
    
    for epoch in range(num_epochs): 
        if verbose:
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 20)
            
        for phase in trainVal: 
            if phase == 'train': 
                imageLoader = train_loader
            else:
                imageLoader = validation_loader
            print('Phase {}'.format(phase))
        
            cur_loss = 0
            cur_correct = 0

            for sample in imageLoader: 
                x_input = sample['x']
                y_true = sample['y']
                
                x_input = Variable(x_input).type(torch.FloatTensor)
                y_true = Variable(y_true).type(torch.LongTensor)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                y_pred = model(x_input)
                _, preds = torch.max(y_pred.data, 1)
                loss = criterion(y_pred, y_true)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                
                cur_loss += loss.data * x_input.size(0)
                cur_correct += torch.sum(preds == y_true.data).item()
                
                #print("preds, y_true", preds, y_true)
                #print("cur_correct", cur_correct)
                
            epoch_loss = cur_loss / data_sizes[phase]
            epoch_acc = cur_correct / data_sizes[phase]
            #print('Cur_correct {}, data_sizes {}'.format(cur_correct, data_sizes[phase]))

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'train':
                loss_hist['train'].append(epoch_loss)
                accuracy_hist['train'].append(epoch_acc)
            else:
                loss_hist['val'].append(epoch_loss)
                accuracy_hist['val'].append(epoch_acc)


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_accuracy = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                best_loss = epoch_loss

    print('Best Validation Acc: {:4f}'.format(best_accuracy))
    
    return model, loss_hist, accuracy_hist, best_weights