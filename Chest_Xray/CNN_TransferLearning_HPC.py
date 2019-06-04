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


class ChestXrayDataset_TL(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for idx in range(len(self.data_frame)):
            img_name = os.path.join(self.root_dir,
                                    self.data_frame.loc[idx, 'Image Index'])

            image = io.imread(img_name)
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = image[:,:,0]

            image=np.repeat(image[None,...],3,axis=0)

            image_class = self.data_frame.loc[idx, 'Class']

            if self.transform:
                image = self.transform(image)

            sample = {'x': image, 'y': image_class}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.samples[idx]


##### Define train, validation, and test loaders 
#################################################

def GetDataLoader(train_csv, validation_csv, test_csv, root_dir, 
               train_transform, validation_transform, 
               batch_size, shuffle, num_workers): 
    chestXray_TrainData = ChestXrayDataset_TL(csv_file = train_csv,
                                        root_dir=root_dir, transform=train_transform)
    train_loader = DataLoader(chestXray_TrainData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)

    chestXray_ValidationData = ChestXrayDataset_TL(csv_file = validation_csv, 
                                                   root_dir=root_dir, 
                                                   transform = validation_transform)
    validation_loader = DataLoader(chestXray_ValidationData, 
                                   batch_size =batch_size, 
                                   shuffle = shuffle, num_workers = num_workers)


    chestXray_TestData = ChestXrayDataset_TL(csv_file = test_csv, 
                                                   root_dir=root_dir, 
                                                   transform=None)
    test_loader = DataLoader(chestXray_TestData, 
                             batch_size = batch_size, 
                             shuffle = shuffle, num_workers = num_workers)
    
    dataset_sizes = {'train': len(chestXray_TrainData), 'val': len(chestXray_ValidationData)}
    return train_loader, validation_loader, test_loader, dataset_sizes



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

                
                cur_loss += loss.item() * x_input.size()[0]
                cur_correct += torch.sum(preds == y_true).item()
                
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

    model.load_state_dict(best_weights)
    print('Best Validation Acc: {:4f}'.format(best_accuracy))
    
    return model, loss_hist, accuracy_hist, best_weights


train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([256,256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([256,256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ######### HPC Paths ######## 
# excel_path = '/home/nhl256/HW2'
# train_local_csv = os.path.join(excel_path, 
#                              'HW2_trainSet.csv')
# validation_local_csv = os.path.join(excel_path, 
#                               "HW2_validationSet.csv")
# test_local_csv = os.path.join(excel_path, 
#                               "HW2_testSet.csv")

# image_path = '/beegfs/ga4493/data/HW2'
# root_image = os.path.join(image_path ,'images')

# NUM_WORKERS = 4
# BATCH_SIZE = 16
# graph_path = '/home/nhl256/HW2/graphs'

# ######### Local Machine Paths ######## 
excel_path = '/Users/nhungle/Box/Free/Deep Learning in Medicine/Deep-Learning-in-Medicine/HW2/excel_files'
train_local_csv = os.path.join(excel_path, 
                              'train_local.csv')
validation_local_csv = os.path.join(excel_path, 
                              'validation_local.csv')
test_local_csv = os.path.join(excel_path, 
                              'test_local.csv')

image_path = '/Users/nhungle/Box/Free/Deep Learning in Medicine/Deep-Learning-in-Medicine/HW2'
root_image = os.path.join(image_path ,'images')

NUM_WORKERS = 1
BATCH_SIZE = 2
graph_path = '/Users/nhungle/Box/Free/Deep Learning in Medicine/Deep-Learning-in-Medicine/HW2/graphs'


##### Loader Data ######
train_loader, validation_loader, test_loader, dataset_sizes = GetDataLoader(train_csv = train_local_csv, validation_csv=validation_local_csv, test_csv=test_local_csv, 
               root_dir = root_image, train_transform=train_transform, 
                validation_transform=validation_transform, 
               batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

#print('datasetSizes', dataset_sizes)




########## Get the model and train ####### 
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch import optim

model_conv_resnet34 = torchvision.models.resnet34(pretrained=True)
for param in model_conv_resnet34.parameters():
    param.requires_grad = False

number_features = model_conv_resnet34.fc.in_features
model_conv_resnet34.fc = torch.nn.Linear(number_features, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv_resnet34.parameters(), lr=0.01)

model_conv_resnet34.train()  
model_conv_resnet34_ft, resnet34_ft_loss, resnet34_ft_acc, resnet34_ft_weights = train_model(model_conv_resnet34, criterion, optimizer=optimizer, 
                    num_epochs=3, data_sizes = dataset_sizes,
               trainVal = ['train', 'val'], verbose = True)

# Plots
fig, ax = plt.subplots()

for key in resnet34_ft_loss: 
    ax.plot(resnet34_ft_loss[key], label = key)

    
ax.set_title('Train and Validation Loss Curves')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
legend = ax.legend(loc= 'best', shadow=True,
                      bbox_to_anchor = (0.5, 0, 0.5, 0.5), ncol = 1, prop = {'size': 10})


plt.savefig(os.path.join(graph_path ,'LossCurves_ResNet34.png'))


def inference(model_ft,loader):
    use_gpu = None
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
        
    ### WHAT SHOULD BE THE Y_SCORE?
    y_score = [output[1] for output in whole_output]
    return y_score, y_target


def write_list_to_file(filename, my_list):
    with open(filename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)

y_score, y_target = inference(model_conv_resnet34_ft, test_loader)
write_list_to_file(os.path.join(graph_path, 'y_score.txt'), y_score)
write_list_to_file(os.path.join(graph_path, 'y_target.txt'), y_target)









