import torch
from torch import nn

class CNN_Disease(nn.Module):
    def __init__(self, out_features=2):
        super(CNN_Disease, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=2)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,32,kernel_size=3,stride=2)
        self.relu4 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.relu6 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.relu7 = nn.ReLU()
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.relu9 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64,128,kernel_size=3,stride=2)
        self.relu10 = nn.ReLU()
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(128,256,kernel_size=3,stride=2)
        self.relu11 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256,out_features)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max1(x)
        y = self.relu2(self.conv2(self.bn1(x)))
        y = self.conv3(y)
        x = self.relu3(y + x)
        x = self.relu4(self.conv4(self.bn2(x)))
        x = self.max2(x)
        y = self.relu5(self.conv5(self.bn3(x)))
        y = self.conv6(y)
        x = self.relu6(y + x)
        x = self.relu7(self.conv7(self.bn4(x)))
        x = self.max3(x)
        y = self.relu8(self.conv8(self.bn5(x)))
        y = self.conv9(y)
        x = self.relu9(y + x)
        x = self.relu10(self.conv10(self.bn6(x)))
        x = self.max4(x)
        x = self.avgpool(self.relu11(self.conv11(x)))
        x = self.fc(x.view(-1,256))
        return x