import os 
import sys 
sys.path.append(os.getcwd())

import torch 
import torch.nn as nn
from torchvision import models
from src.statistic import RunningMean

import pytorch_lightning as pl

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
        
class DogCatModel(pl.LightningModule):
    def __init__(self, model, lr = 2e-4):
        super().__init__()
        if model == 'cnn':
            self.model = CNN()
    
        self.train_loss = RunningMean()
        self.val_loss   = RunningMean()
        self.train_acc  = RunningMean()
        self.val_acc    = RunningMean()
        
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)

    def _cal_loss_and_acc(self, batch):
        """
            This method is used to calculate loss and accuracy for each batch.
        """
        x, y = batch
        y_hat = self(x) # Forward pass
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean() # 
        return loss,acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._cal_loss_and_acc(batch)
        self.train_loss.update(loss.item(), batch[0].shape[0]) # batch[0] here is the batch size and batch[1] is the label, so batch[0].shape[0] is the batch size
        self.train_acc.update(acc.item(), batch[0].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss,acc = self._cal_loss_and_acc(batch)
        self.val_loss.update(loss.item(),batch[0].shape[0])
        self.val_acc.update(acc.item(),batch[0].shape[0])
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_loss",self.train_loss(),sync_dist=True)
        self.log("train_acc",self.train_acc(),sync_dist=True)
        
        # Reset the metrics
        self.train_loss.reset()
        self.train_acc.reset()
    
    def on_validation_epoch_end(self):
        self.log("val_loss",self.val_loss(),sync_dist=True)
        self.log("val_acc",self.val_acc(),sync_dist=True)
        
        # Reset the metrics
        self.val_loss.reset()
        self.val_acc.reset()
    
    def test_step(self,batch,batch_idx):
        loss, acc = self._cal_loss_and_acc(batch)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

if __name__=='__main__':
    cnn = CNN()
    print(f"CNN Output Shape: {cnn(torch.randn(32,3,112,112)).shape}")
    