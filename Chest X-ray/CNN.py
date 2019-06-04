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


class ChestXrayDataset(Dataset):
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

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.loc[idx, 'Image Index'])

        image = io.imread(img_name,as_gray=True)
        
        image = (image - image.mean()) / image.std()
            
        image_class = self.data_frame.loc[idx, 'Class']

        sample = {'x': image[None,:], 'y': image_class}

        if self.transform:
            sample = self.transform(sample)

        return sample

def GetDataLoader(train_csv, validation_csv, test_csv, root_dir, 
               train_transform, validation_transform, 
               batch_size, shuffle, num_workers): 
    chestXray_TrainData = ChestXrayDataset(csv_file = train_csv,
                                        root_dir=root_dir, transform=train_transform)
    train_loader = DataLoader(chestXray_TrainData, batch_size=batch_size,
                            shuffle = shuffle, num_workers = num_workers)

    chestXray_ValidationData = ChestXrayDataset(csv_file = validation_csv, 
                                                   root_dir=root_dir, 
                                                   transform = validation_transform)
    validation_loader = DataLoader(chestXray_ValidationData, 
                                   batch_size =batch_size, 
                                   shuffle = shuffle, num_workers = num_workers)


    chestXray_TestData = ChestXrayDataset(csv_file = test_csv, 
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
# graph_path = '/home/nhl256/HW2'


##### Loader Data ######
train_loader, validation_loader, test_loader, dataset_sizes = GetDataLoader(train_csv = train_local_csv, validation_csv=validation_local_csv, test_csv=test_local_csv, 
               root_dir = root_image, train_transform=None, 
                validation_transform=None, 
               batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


convresnet_model = Conv_ResNet()
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCELoss()
optimizer = optim.SGD(convresnet_model.parameters(), lr=0.01)

convresnet_model.train()  
convresnet_model_ft, convresnet_model_ft_loss, convresnet_model_ft_acc, convresnet_model_ft_weights = train_model(convresnet_model, criterion, optimizer=optimizer, 
                    num_epochs=5, data_sizes = dataset_sizes,
               trainVal = ['train', 'val'], verbose = True)


# Plots
fig, ax = plt.subplots()

for key in convresnet_model_ft_loss: 
    ax.plot(convresnet_model_ft_loss[key], label = key)

    
ax.set_title('Train and Validation Loss Curves')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
legend = ax.legend(loc= 'best', shadow=True,
                      bbox_to_anchor = (0.5, 0, 0.5, 0.5), ncol = 1, prop = {'size': 10})


plt.savefig(os.path.join(graph_path ,'LossCurves_convresnet.png'))

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

y_score, y_target = inference(convresnet_model_ft, train_loader)
write_list_to_file(os.path.join(graph_path, 'Q4_y_score.txt'), y_score)
write_list_to_file(os.path.join(graph_path, 'Q4_y_target.txt'), y_target)



