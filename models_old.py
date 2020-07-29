"""### Model"""

import torch.nn as nn
import torch.nn.functional as F


class myDeepCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        # set depth
        self.depth = 2

        # convolutional layers
        self.conv_features_1 = nn.Sequential(
        # first convolutional set of layers
            nn.Conv2d(in_channels=176, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True))

        self.conv_features_2 = nn.Sequential(
          # second convolutional set of layers
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64))

        self.conv_features_3 = nn.Sequential(
          #bottleneck layer to reduce size to be fed to classifier
            nn.Conv2d(64, 32, 5, padding=2),
            nn.MaxPool2d(3,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(3,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1),
            nn.MaxPool2d(3,1),
            nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(
          nn.Flatten(),
          # size imposed by adaptive pooling
          nn.Linear(288, 256),
          nn.Linear(256, num_classes))

        self.conv_features = str(self.depth)+"-skip connected layer"

    def forward(self, x):
        x = self.conv_features_1(x)
        x = self.conv_features_2(x)
        residual = x.clone()
        # skip connection 1
        for i in range(self.depth):
            x = self.conv_features_2(x)
            x, residual = self.skip_connection(x, residual)
        x = self.conv_features_2(x)
        x = self.conv_features_3(x)
        x = self.classifier(x)
        return x

    def skip_connection(self, x, residual):
        x += residual
        # Relu needs to be after the skip connection addition
        x = F.relu(x)
        residual = x.clone()
        return x, residual


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class CNN(nn.Module):
    def __init__(self, image_channels=176, h_dim=128, z_dim=32, num_classes=14):
        super().__init__()
        self.conv_features1 = nn.Sequential(
            nn.Conv2d(image_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())
        
        self.conv_features2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU())
            
        self.conv_features3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU())
        # for printing purposes
        self.conv_features = nn.Sequential(
            self.conv_features1,
            self.conv_features2,
            self.conv_features3
            )

        self.flatten = Flatten() 
        
        self.unflatten = UnFlatten()
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)
        self.maxunpool3 = nn.MaxUnpool2d(kernel_size=2)
        
        self.deconv_features1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU())
            
        self.deconv_features2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        self.deconv_features3 = nn.Sequential(
            nn.ConvTranspose2d(256, image_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        self.classifier = nn.Sequential(
          nn.Linear(16384, 64),
          nn.Linear(64, num_classes))

        # self.fc1 = nn.Linear(h_dim, h_dim//2)
        # self.fc2 = nn.Linear(h_dim//2, z_dim)
        # self.fc3 = nn.Linear(z_dim, h_dim//2)
        # self.fc4 = nn.Linear(h_dim//2, )

        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        x = self.conv_features1(x)
        self.size1 = x.size()
        x, self.indices1 =self.maxpool1(x)
        x = self.conv_features2(x)
        self.size2 = x.size()
        x, self.indices2 =self.maxpool2(x)
        x = self.conv_features3(x)
        self.size3 = x.size()
        x, self.indices3 =self.maxpool3(x)
        h = self.flatten(x)
        return h
    
    def classify(self, h):
        return(classifier(h))

    def decode(self, h):
        z = self.unflatten(h)
        z = self.deconv_features1(x)
        z =self.maxunpool1(x, self.indices3, self.size3)
        z = self.deconv_features2(x)
        z =self.maxunpool2(x, self.indices2, self.size2)
        z = self.deconv_features3(x)
        z =self.maxunpool3(x, self.indices1, self.size1)
        z = self.sigmoid(z)
        return z

    def forward(self, x):
        h = self.encode(x)
        c = self.classifier(h)
        #z = self.decode(z)
        return c

# class CNN(nn.Module):
#     def __init__(self, image_channels=176, h_dim=128, z_dim=32):
#         super(CNN, self).__init__()
#     def forward(self, x):
#       pass
