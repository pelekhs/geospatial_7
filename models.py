"""### Model"""
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# class UnFlatten(nn.Module):
#     def forward(self, input, size=1024, size2=1):
#         return input.view(input.size(0), size, size2, size2)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class old_AE(nn.Module):
    def __init__(self, image_channels=176, num_classes=14, supervision=True):
        super().__init__()
        
        self.supervision = supervision
        self.linear_size = 9216
        
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

        # self.flatten = Flatten()

        # self.unflatten = UnFlatten()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.maxUnpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.maxUnpool2 = nn.MaxUnpool2d(kernel_size=2)
        self.maxUnpool3 = nn.MaxUnpool2d(kernel_size=2)

        self.deconv_features1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU())

        self.deconv_features2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        self.deconv_features3 = nn.Sequential(
            nn.ConvTranspose2d(256, image_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=image_channels),
            nn.ReLU())

        self.classifier = nn.Sequential(
          nn.Linear(self.linear_size, 64),
          nn.Linear(64, num_classes))

        self.sigmoid = nn.Sigmoid()

        #init weights
        if not self.supervision:
            self.conv_features1.apply(init_weights)
            self.conv_features2.apply(init_weights)
            self.conv_features3.apply(init_weights)
            self.deconv_features1.apply(init_weights)
            self.deconv_features2.apply(init_weights)
            self.deconv_features3.apply(init_weights)
        self.classifier.apply(init_weights)

    def encode(self, x):
        x = self.conv_features1(x)
        self.size1 = x.size()
        x, self.indices1 =self.maxpool1(x)
        x = self.conv_features2(x)
        self.size2 = x.size()
        x, self.indices2 =self.maxpool2(x)
        x = self.conv_features3(x)
        self.size3 = x.size()
        h, self.indices3 =self.maxpool3(x)
        #h = self.flatten(x)
        return h

    def classify(self, h):
        return(self.classifier(h))

    def decode(self, h):
        #z = self.unflatten(h, 1024, 3)
        z = self.maxUnpool1(h, self.indices3, self.size3)
        z = self.deconv_features1(z)
        z = self.maxUnpool2(z, self.indices2, self.size2)
        z = self.deconv_features2(z)
        z = self.maxUnpool3(z, self.indices1, self.size1)
        z = self.deconv_features3(z)
        z = self.sigmoid(z)
        return z

    def forward(self, x):
        h = self.encode(x)
        if self.supervision:
            return self.classify(h)
        else:
            return self.decode(h)

class AE(nn.Module):
    def __init__(self, image_channels=176, num_classes=14, supervision=True):
        super().__init__()

        div = 2
        pool_stride = 2
        pool_kernel = 2
        
        self.supervision = supervision

        self.conv_features1 = nn.Sequential(
            nn.Conv2d(image_channels, 64 // div, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64 // div),
            nn.ReLU(),
            nn.Conv2d(64 // div, 64 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64 // div),
            nn.ReLU(),
            # downscaling layer to original size for the skip connection
            nn.Conv2d(64 // div, image_channels, kernel_size=1, padding=0)
            )

        self.conv_features2 = nn.Sequential(
            nn.Conv2d(image_channels, 128 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128// div),
            nn.ReLU(),
            nn.Conv2d(128 // div, 128 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128 // div),
            nn.ReLU(),
            nn.Conv2d(128 // div, image_channels, kernel_size=1, padding=0)
            )

        self.conv_features3 = nn.Sequential(
            nn.Conv2d(image_channels, 256 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256 // div),
            nn.ReLU(),
            nn.Conv2d(256 // div, 256 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256 // div),
            nn.ReLU(),
            nn.Conv2d(256 // div, image_channels, kernel_size=1, padding=0)
            )


        self.flatten = Flatten()
        # self.unflatten = UnFlatten()

        self.maxpool1 = nn.MaxPool2d(kernel_size=pool_kernel,
                                     stride=pool_stride, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=pool_kernel, 
                                     stride=pool_stride, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=pool_kernel,
                                     stride=pool_stride,return_indices=True)

        self.maxUnpool1 = nn.MaxUnpool2d(kernel_size=pool_kernel, 
                                         stride=pool_stride)
        self.maxUnpool2 = nn.MaxUnpool2d(kernel_size=pool_kernel, 
                                         stride=pool_stride)
        self.maxUnpool3 = nn.MaxUnpool2d(kernel_size=pool_kernel, 
                                         stride=pool_stride)

        self.deconv_features1 = nn.Sequential(
            # upscale to the desired dimension
            nn.ConvTranspose2d(image_channels, 256 // div, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=256 // div),
            nn.ReLU(),
            nn.ConvTranspose2d(256 // div, 256 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256 // div),
            nn.ReLU(),
            nn.ConvTranspose2d(256 // div, image_channels, kernel_size=3, padding=1),
            )

        self.deconv_features2 = nn.Sequential(
            nn.Conv2d(image_channels, 128 // div, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=128 // div),
            nn.ReLU(),
            nn.Conv2d(128 // div, 128 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128 // div),
            nn.ReLU(),
            nn.Conv2d(128 // div, image_channels, kernel_size=3, padding=1))

        self.deconv_features3 = nn.Sequential(
            nn.Conv2d(image_channels, 64 // div, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64 // div),
            nn.ReLU(),
            nn.Conv2d(64 // div, 64 // div, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64 // div),
            nn.ReLU(),
            nn.Conv2d(64 // div, image_channels, kernel_size=3, padding=1)
            )
        
        self.classifier = nn.Sequential(
          nn.Linear(704, 1024),
          nn.Dropout(p=0.2),
          nn.Linear(1024, 256),
          nn.Linear(256, num_classes))
        
        # for printing purposes
        self.conv_features = nn.Sequential(
            self.conv_features1,
            self.maxpool1,
            self.conv_features2,
            self.maxpool2,
            self.conv_features3,
            self.maxpool3
            )

        self.sigmoid = nn.Sigmoid()

        #init weights
        if not self.supervision:
            self.conv_features1.apply(init_weights)
            self.conv_features2.apply(init_weights)
            self.conv_features3.apply(init_weights)
            self.deconv_features1.apply(init_weights)
            self.deconv_features2.apply(init_weights)
            self.deconv_features3.apply(init_weights)
        self.classifier.apply(init_weights)

    def encode(self, x):
        residual = x.clone()
        x = self.conv_features1(x)
        self.size1 = x.size()
        x += residual
        x, self.indices1 =self.maxpool1(x)

        residual = x.clone()
        x = self.conv_features2(x)
        self.size2 = x.size()
        x += residual
        x, self.indices2 =self.maxpool2(x)

        residual = x.clone()
        x = self.conv_features3(x)
        self.size3 = x.size()
        x += residual
        h, self.indices3 =self.maxpool3(x)
        return h

    def decode(self, h):
        z = self.maxUnpool1(h, self.indices3, self.size3)
        residual = z.clone()
        z = self.deconv_features1(z)
        z += residual
        z =self.maxUnpool2(z, self.indices2, self.size2)
        residual = z.clone()
        z = self.deconv_features2(z)
        z += residual
        z =self.maxUnpool3(z, self.indices1, self.size1)
        residual = z.clone()
        z = self.deconv_features3(z)
        z += residual
        z = self.sigmoid(z)
        return z


    def classify(self, h):
        return(self.classifier(h))

    def forward(self, x):
        h = self.encode(x)
        if self.supervision:
            return self.classify(self.flatten(h))
        else:
            return self.decode(h)
        
class Classifier(nn.Module):
  def __init__(self, num_classes=14, dropout=0.0):
    super().__init__()
    self.dropout = dropout
    self.flatten = Flatten()
    self.num_classes = num_classes
    self.classifier = nn.Sequential(
        nn.Linear(176, 2048),
        nn.Dropout(p=self.dropout),
    #    nn.Linear(512, 512),
    #    nn.Linear(512, 256),
    #  nn.Linear(256, 128),
        nn.Linear(2048, num_classes))
      
  def forward(self, x):
    x = self.flatten(x)
    x = self.classifier(x)
    return x

class AE3D(nn.Module):
    def __init__(self, image_channels=176, num_classes=14, supervision=True):
        super().__init__()

        div = 1
        kernel_depth = 2
        pool_stride = 3
        pool_kernel = 3
        
        self.supervision = supervision
        self.image_channels = image_channels
        self.kernel_depth = kernel_depth
        
        self.conv_features1 = nn.Sequential(
            nn.Conv3d(image_channels // kernel_depth, 64 // div, kernel_size=
                      (kernel_depth,3,3), stride=1, padding=1),
            nn.BatchNorm3d(num_features = 64 // div),
            nn.LeakyReLU(),
            nn.Conv3d(64 // div, 64 // div, kernel_size=
                      (kernel_depth,3,3), padding=1),
            nn.BatchNorm3d(num_features = 64 // div),
            nn.LeakyReLU(),
            # downscaling layer to original size for the skip connection
            nn.Conv3d(64 // div, image_channels // kernel_depth, kernel_size=1, padding=0)
            )

        self.conv_features2 = nn.Sequential(
            nn.Conv3d(image_channels, 128 // div, kernel_size=(kernel_depth,3,3), padding=1),
            nn.BatchNorm3d(num_features = 128 // div),
            nn.LeakyReLU(),
            nn.Conv3d(128 // div, 128 // div, kernel_size=(kernel_depth,3,3), padding=1),
            nn.BatchNorm3d(num_features = 128 // div),
            nn.LeakyReLU(),
            nn.Conv3d(128 // div, image_channels // kernel_depth, kernel_size=1, padding=0)
            )

        self.conv_features3 = nn.Sequential(
            nn.Conv3d(image_channels, 256 // div, kernel_size=(kernel_depth,3,3), padding=1),
            nn.BatchNorm3d(num_features = 256 // div),
            nn.LeakyReLU(),
            nn.Conv3d(256 // div, 256 // div, kernel_size=(kernel_depth,3,3), padding=1),
            nn.BatchNorm3d(num_features = 256 // div),
            nn.LeakyReLU(),
            nn.Conv3d(256 // div, image_channels // kernel_depth, kernel_size=1, padding=0)
            )


        self.flatten = Flatten()
        # self.unflatten = UnFlatten()

        self.maxpool1 = nn.MaxPool3d(kernel_size=pool_kernel,
                                     stride=pool_stride, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=pool_kernel, 
                                     stride=pool_stride, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=pool_kernel,
                                     stride=pool_stride,return_indices=True)

        self.maxUnpool1 = nn.MaxUnpool3d(kernel_size=pool_kernel, 
                                         stride=pool_stride)
        self.maxUnpool2 = nn.MaxUnpool3d(kernel_size=pool_kernel, 
                                         stride=pool_stride)
        self.maxUnpool3 = nn.MaxUnpool3d(kernel_size=pool_kernel, 
                                         stride=pool_stride)

        self.deconv_features1 = nn.Sequential(
            # upscale to the desired dimension
            nn.ConvTranspose2d(image_channels, 256 // div, kernel_size=1, 
                               padding=0),
            nn.BatchNorm3d(num_features = 256 // div),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256 // div, 256 // div, kernel_size=
                               (kernel_depth,3,3), padding=1),
            nn.BatchNorm3d(num_features = 256 // div),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256 // div, image_channels // kernel_depth,
                               kernel_size=(kernel_depth,3,3), padding=1),
            )

        self.deconv_features2 = nn.Sequential(
            nn.Conv3d(image_channels, 128 // div, kernel_size=1, padding=0),
            nn.BatchNorm3d(num_features = 128 // div),
            nn.LeakyReLU(), 
            nn.Conv3d(128 // div, 128 // div, kernel_size=(kernel_depth,3,3),
                      padding=1),
            nn.BatchNorm3d(num_features = 128 // div),
            nn.LeakyReLU(),
            nn.Conv3d(128 // div, image_channels // kernel_depth, 
                      kernel_size=(kernel_depth,3,3), padding=1))

        self.deconv_features3 = nn.Sequential(
            nn.Conv3d(image_channels, 64 // div, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm3d(num_features = 64 // div),
            nn.LeakyReLU(),
            nn.Conv3d(64 // div, 64 // div, kernel_size=(kernel_depth,3,3),
                      padding=1),
            nn.BatchNorm3d(num_features = 64 // div),
            nn.LeakyReLU(),
            nn.Conv3d(64 // div, image_channels // kernel_depth, kernel_size=
                      (kernel_depth,3,3), padding=1)
            )
        
        self.classifier = nn.Sequential(
          nn.Linear(704, 1024),
          nn.Dropout(p=0.2),
          nn.Linear(1024, 256),
          nn.Linear(256, num_classes))
        
        # for printing purposes
        self.conv_features = nn.Sequential(
            self.conv_features1,
            self.maxpool1,
            self.conv_features2,
            self.maxpool2,
            self.conv_features3,
            self.maxpool3
            )

        self.sigmoid = nn.Sigmoid()

        #init weights
        if not self.supervision:
            self.conv_features1.apply(init_weights)
            self.conv_features2.apply(init_weights)
            self.conv_features3.apply(init_weights)
            self.deconv_features1.apply(init_weights)
            self.deconv_features2.apply(init_weights)
            self.deconv_features3.apply(init_weights)
        self.classifier.apply(init_weights)

    def encode(self, x):
        x = x.resize(x.size(0), x.size(1) // self.kernel_depth, self.kernel_depth,
                     x.size(2), x.size(3))
        residual = x.clone()
        print(residual.size())
        x = self.conv_features1(x)
        self.size1 = x.size()
        print(x.size())
        x += residual
        x, self.indices1 =self.maxpool1(x)

        residual = x.clone()
        x = self.conv_features2(x)
        self.size2 = x.size()
        x += residual
        x, self.indices2 =self.maxpool2(x)

        residual = x.clone()
        x = self.conv_features3(x)
        self.size3 = x.size()
        x += residual
        h, self.indices3 =self.maxpool3(x)
        return h

    def classify(self, h):
        return(self.classifier(h))

    def decode(self, h):
        z = self.maxUnpool1(h, self.indices3, self.size3)
        residual = z.clone()
        z = self.deconv_features1(z)
        z += residual
        z =self.maxUnpool2(z, self.indices2, self.size2)
        residual = z.clone()
        z = self.deconv_features2(z)
        z += residual
        z =self.maxUnpool3(z, self.indices1, self.size1)
        residual = z.clone()
        z = self.deconv_features3(z)
        z += residual
        z = self.sigmoid(z)
        return z

    def forward(self, x):
        h = self.encode(x)
        if self.supervision:
            return self.classify(self.flatten(h))
        else:
            return self.decode(h)
    
