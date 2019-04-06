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

def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs = 10,verbose = True):

    start_time = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    acc_dict = {'train':[],'validation':[]}
    loss_dict = {'train':[],'validation':[]}

    for epoch in range(num_epochs):
        if verbose:
            #if epoch % 5 == 4:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
            
        for phase in ['train','val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                
                inputs = data['x']
                labels = data['y']
                
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.to(device)
                else:
                    inputs = Variable(inputs).type(torch.FloatTensor)
                    labels = Variable(labels).type(torch.LongTensor)

                    
                optimizer.zero_grad()

                out = model(inputs)
                _, preds = torch.max(out, dim = 1)
                loss = criterion(out, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * inputs.size()[0]
                running_corrects += torch.sum(preds == labels).item()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if verbose:
#                 if epoch % 5 == 4:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                loss_dict['train'].append(epoch_loss)
                acc_dict['train'].append(epoch_acc)
            else:
                loss_dict['validation'].append(epoch_loss)
                acc_dict['validation'].append(epoch_acc)
                    
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                scheduler.step(epoch_loss)

    time_elapsed = time.time() - start_time
    print('Training time: {}minutes {}s'.format(int(time_elapsed / 60), time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # for i, phase in enumerate(['train','validation']):

    #     fig = plt.figure()
        
    #     a = fig.add_subplot(1,2,1*i+1)
    #     plt.plot(loss_dict[phase])
    #     plt.title('Loss per epoch for ' + phase)

    #     a = fig.add_subplot(1,2,1*i+2)
    #     plt.plot(acc_dict[phase])
    #     plt.title('Accuracy per epoch for ' + phase)
    #     plt.show()
    #     plt.savefig(os.path.join(graph_path ,'Curve {}.png'.format(phase)))

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(graph_path, '{}.pt'.format(model_name)))
    
    return {'Model': model, 'LossDict': loss_dict, 'AccDict': acc_dict}