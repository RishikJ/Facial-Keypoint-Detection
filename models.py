## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        #convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2=nn.Conv2d(32,64,3)
        self.conv3=nn.Conv2d(64,128,3)
        self.conv4=nn.Conv2d(128,256,3)
        self.conv5=nn.Conv2d(256,512,1)
        
        #dropout layers
        self.drop1=nn.Dropout(p=0.1)
        self.drop2=nn.Dropout(p=0.2)
        self.drop3=nn.Dropout(p=0.25)
        self.drop4=nn.Dropout(p=0.25)
        self.drop5=nn.Dropout(p=0.3)
        self.drop6=nn.Dropout(p=0.4)
        
        #Batchnorm layers
        #self.batch1=nn.BatchNorm2d(32)
        #self.batch2=nn.BatchNorm2d(64)
        #self.batch3=nn.BatchNorm2d(128)
        #self.batch4=nn.BatchNorm2d(256)
        
        #pooling layers
        self.pool=nn.MaxPool2d(2,2)
        
        #linear layers
        self.fc1=nn.Linear(512*6*6, 1024, bias=True)
        self.fc2=nn.Linear(1024, 136, bias=True)
        
        
        


        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        x = self.drop5(self.pool(F.relu(self.conv5(x))))
        
        
        #flattening the layers
        x=x.view(x.size(0),-1)
        
        #Fully connected layers
        x=self.drop6(F.relu(self.fc1(x)))
        x=self.fc2(x)
        
        
        
        
        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
