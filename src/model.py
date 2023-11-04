import os
import sys
sys.path.append(os.getcwd())  # NOQA

import torch
import pytorch_lightning as pl
from src.statistic import RunningMean
from torchvision import models
import torch.nn as nn


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
        # Flatten the output of conv layers to feed into FC layers. Here, x.size(0) is the batch size
        x = x.view(x.size(0), -1)
        x = self.fc1(x)                 # 512
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                 # 2 (2 classes)

        return x

    def _get_kernel(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


class Depthwise_Conv(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(Depthwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels  # Conv on each channel separately, then stack the results
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class Pointwise_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Depthwise_Separable_Conv, self).__init__()
        self.dw = Depthwise_Conv(in_channels=in_channels, stride=stride)
        self.pw = Pointwise_Conv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, input_image):
        x = self.pw(self.dw(input_image))
        return x


class MobileNetV2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = models.mobilenet_v2(pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x


class MobileNetV3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = models.mobilenet_v3_small(pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # Conv / s2
            nn.BatchNorm2d(32),
            nn.ReLU(),

            Depthwise_Separable_Conv(32, 64, stride=1),  # Conv dw / s1
            Depthwise_Separable_Conv(64, 128, stride=2),  # Conv dw / s2
            Depthwise_Separable_Conv(128, 128, stride=1),  # Conv dw / s1
            Depthwise_Separable_Conv(128, 256, stride=2),  # Conv dw / s2
            Depthwise_Separable_Conv(256, 256, stride=1),  # Conv dw / s1
            Depthwise_Separable_Conv(256, 512, stride=2),  # Conv dw / s2

            # 5x Conv dw / s1
            Depthwise_Separable_Conv(512, 512, stride=1),
            Depthwise_Separable_Conv(512, 512, stride=1),
            Depthwise_Separable_Conv(512, 512, stride=1),
            Depthwise_Separable_Conv(512, 512, stride=1),
            Depthwise_Separable_Conv(512, 512, stride=1),

            # 2x Conv dw / s2
            Depthwise_Separable_Conv(512, 1024, stride=2),
            Depthwise_Separable_Conv(1024, 1024, stride=2),

            # AvgPool
            nn.AdaptiveAvgPool2d(1),  # AvgPool / s1
        )

        self.fc = nn.Linear(1024, num_classes)  # FC / s1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        print(x.shape)
        x = x.view(x.size(0), 1024)
        print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = models.resnet18(pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x


class DogCatModel(pl.LightningModule):
    def __init__(self, model, lr=2e-4):
        super().__init__()
        if model == 'cnn':
            self.model = CNN()
        elif model == 'mobilenetv1':
            self.model = MobileNetV1(3, 2)
        elif model == 'mobilenetv2':
            self.model = MobileNetV2()
        elif model == 'mobilenetv3':
            self.model = MobileNetV3()

        self.train_loss = RunningMean()
        self.val_loss = RunningMean()
        self.train_acc = RunningMean()
        self.val_acc = RunningMean()

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _cal_loss_and_acc(self, batch):
        """
            This method is used to calculate loss and accuracy for each batch.
        """
        x, y = batch
        y_hat = self(x)  # Forward pass
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._cal_loss_and_acc(batch)
        # batch[0] here is the batch size and batch[1] is the label, so batch[0].shape[0] is the batch size
        self.train_loss.update(loss.item(), batch[0].shape[0])
        self.train_acc.update(acc.item(), batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._cal_loss_and_acc(batch)
        self.val_loss.update(loss.item(), batch[0].shape[0])
        self.val_acc.update(acc.item(), batch[0].shape[0])
        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss(), sync_dist=True)
        self.log("train_acc", self.train_acc(), sync_dist=True)

        # Reset the metrics
        self.train_loss.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss(), sync_dist=True)
        self.log("val_acc", self.val_acc(), sync_dist=True)

        # Reset the metrics
        self.val_loss.reset()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        loss, acc = self._cal_loss_and_acc(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    cnn = CNN()
    print(f"CNN Output Shape: {cnn(torch.randn(32,3,112,112)).shape}")
