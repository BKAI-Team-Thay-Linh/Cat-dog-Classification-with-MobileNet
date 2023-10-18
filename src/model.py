import os 
import sys 
sys.path.append(os.getcwd())

import torch 
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = self._get_kernel(3, 64)
        self.conv2 = self._get_kernel(64, 128)
        self.conv3 = self._get_kernel(128, 256)
        self.fc1 = nn.Linear(256*14*14, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)               # 112x112
        x = self.relu(x)                # 112x112
        x = nn.MaxPool2d(2, 2)(x)       # 56x56 (112/2)
        x = self.conv2(x)               # 56x56 
        x = self.relu(x)                # 56x56
        x = nn.MaxPool2d(2, 2)(x)       # 28x28 (56/2)
        x = self.conv3(x)               # 28x28
        x = self.relu(x)                # 28x28
        x = nn.MaxPool2d(2, 2)(x)       # 14x14 (28/2)
        
        # Fully connected layers
        x = x.view(x.size(0), -1)       # Flatten the output of conv layers to feed into FC layers. Here, x.size(0) is the batch size
        x = x.unsqueeze(0)              # Add a dimension of size 1 at index 0
        x = self.fc1(x)                 # 512
        x = self.relu(x)                # 512
        x = self.dropout(x)             
        x = self.fc2(x)                 # 2 (2 classes)
        x = self.softmax(x)

        return x
    
    def _get_kernel(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

if __name__=='__main__':
    cnn = CNN()
    print(f"CNN Output Shape: {cnn(torch.randn(32,3,112,112)).shape}")
    