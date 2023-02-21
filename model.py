import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNNet(nn.Module):
    def __init__(self,num_class):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),
                      padding=3,stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      padding=1, stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                      padding=1, stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                      padding=1, stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3),
                      padding=1, stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU6(),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU6(),
            nn.Linear(128,num_class)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(x.shape)
        x = F.avg_pool2d(x,kernel_size = (x.shape[2],x.shape[3]))
        # print(x.shape)
        x = x.reshape(-1,x.shape[1])
        # print(x.shape)
        x =self.fc(x)
        return x


#===============================
    # def __init__(self, num_class):
    #     super(CNNNet, self).__init__()
    #     # conv1
    #     self.conv1 = nn.Sequential(
    #         nn.Conv2d(in_channels=num_class, out_channels=32, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #         nn.LocalResponseNorm(size=5)
    #     )
    #     # conv2
    #     self.conv2 = nn.Sequential(
    #         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #         nn.LocalResponseNorm(size=5)
    #     )
    #     # conv3
    #     self.conv3 = nn.Sequential(
    #         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU()
    #     )
    #     # conv4
    #     self.conv4 = nn.Sequential(
    #         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU()
    #     )
    #     # conv5
    #     self.conv5 = nn.Sequential(
    #         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(512),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2)
    #
    #
    #     )
    #     self.fc1 = nn.Linear(512 , 128)
    #     self.fc2 = nn.Linear(128, 3)
    #
    #
    # def forward(self, input):
    #     # print(input.size())
    #     out = self.conv1(input)
    #     # print(out.size())
    #     out = self.conv2(out)
    #     # print(out.size())
    #     out = self.conv3(out)
    #     # print(out.size())
    #     out = self.conv4(out)
    #     # print(out.size())
    #     out = self.conv5(out)
    #     print(out.size())
    #     out = F.avg_pool2d(out, kernel_size=(224, 224))
    #     out = out.squeeze()
    #     print(out.size())
    #     out = self.fc1(out)
    #     # print(out.size())
    #     out = self.fc2(out)
    #     # print(out.size())

        #
        # return out
