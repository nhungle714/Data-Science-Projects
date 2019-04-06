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



################ Plot #####################
def PlotAccLoss(model, model_name): 
    fig, ax = plt.subplots()
    for key in model['LossDict']: 
        ax.plot(model['LossDict'][key], label = key)
    ax.set_title('Train and Validation Loss Curves')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    legend = ax.legend(loc= 'best', shadow=True,
                          bbox_to_anchor = (0.5, 0, 0.5, 0.5), ncol = 1, prop = {'size': 10})
    plt.savefig(os.path.join(graph_path ,'LossCurves_{}.png'.format(model_name)))

    fig, ax = plt.subplots()
    for key in model['AccDict']: 
        ax.plot(model['AccDict'][key], label = key)
    ax.set_title('Train and Validation Accuracy Curves')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epochs')
    legend = ax.legend(loc= 'best', shadow=True,
                          bbox_to_anchor = (0.5, 0, 0.5, 0.5), ncol = 1, prop = {'size': 10})
    plt.savefig(os.path.join(graph_path ,'AccuracyCurves_{}.png'.format(model_name)))

################ Evaluation on Test Set #####################
def inference(model_ft,loader):
    model_ft.eval()
    whole_output =[]
    whole_target = []
    

    for valData in loader:
        data = valData['x']
        target = valData['y']
        if use_gpu:
            data = Variable(data,volatile=True).type(torch.FloatTensor).cuda()
            target = Variable(target,volatile=True).type(torch.LongTensor).cuda()
        else:
            data= Variable(data,volatile=True).type(torch.FloatTensor)
            target = Variable(target,volatile=True).type(torch.LongTensor)

        output =F.softmax(model_ft(data),dim=1)
        whole_output.append( output.cpu().data.numpy())
        whole_target.append( valData['y'].numpy())

    whole_output = np.concatenate(whole_output)
    whole_target = list(np.concatenate(whole_target))
    y_target = whole_target


    #print('Whole_output: {}, whole_target: {}'.format(whole_output, whole_target))
    #print('y_target: {}'.format(y_target))
    y_score = [output[1] for output in whole_output]
    return y_score, y_target

def write_list_to_file(filename, my_list):
    with open(filename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)
