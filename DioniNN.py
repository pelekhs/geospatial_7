"""Data loading"""

from import_images import *

"""Data preparation for patch based classification
Batch cropping functions
Combines padding where necessary along with square cropping of variable 
size around labeled pixels. This process creates a new image training set 
which provides a patch-based classification method for each labeled pixel. 
Every patch will be classified according to the label of its central pixel.
"""
from batch_crop import *

"""CUDA"""

import torch

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.empty_cache()
    device = torch.device("cuda") 
    print("\nGPU is available")
else:
    device = torch.device("cpu")
    print("\nGPU not available, CPU used")

"""Dataset"""

from dataset import *

"""Training tools"""
from training import trainNN

"""Settings and hyperparameters"""
from config import *

"""Models"""
from models import *
model = models[selected_model].to(device)
if pre_model:
    pre_model.to(device)

from torchsummary import summary
if not pretrained_encoder:
    print(summary(model, (176, patch_size, patch_size)))

"""Train / Test split
In this section, patches of selected dimensions are created and split into train / test / validation sets. Firstly, it is essential to set exclusive subsets of each image to functions as banks of train / test / validation patches so that the respective patches are prevented from overlaps. Overlaps mean cheating that is bad generalisation to the test images. It was considered preferable to split the images horizontally given that such a split retains in the best possible degree the existence of all classes in each set.
"""

if not train_all:
    
    train = [dioni[:, dioni.shape[1]//4:, :], loukia[:, loukia.shape[1]//4:, :]]
    
    train_gt = [dioni_gt[:, dioni_gt.shape[1]//4:, :], 
                loukia_gt[:, loukia_gt.shape[1]//4:, :]]
    
else: # train over all of dioni and loukia
    
    train = [dioni, loukia]
    
    train_gt = [dioni_gt, loukia_gt]
    
    
val = [dioni[:, dioni.shape[1]//8:dioni.shape[1]//4, :],
         loukia[:, loukia.shape[1]//8:loukia.shape[1]//4, :]]

test = [dioni[:, :dioni.shape[1]//8, :],
         loukia[:, :loukia.shape[1]//8, :]]


val_gt = [dioni_gt[:, dioni_gt.shape[1]//8:dioni_gt.shape[1]//4, :],
         loukia_gt[:, loukia_gt.shape[1]//8:loukia_gt.shape[1]//4, :]]

test_gt = [dioni_gt[:, :dioni_gt.shape[1]//8, :],
         loukia_gt[:, :loukia_gt.shape[1]//8, :]]

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test= []
coords = []

if supervision:
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # patch training set
    for (img, gt) in zip(train, train_gt):
        patch, label = patches_supervised(img=img, gt=gt, size=patch_size)
        X_train.extend(patch)
        y_train.extend(label)
    
    # patch validation set
    for (img, gt) in zip(val, val_gt):
        patch, label = patches_supervised(img=img, gt=gt, size=patch_size)
        X_val.extend(patch)
        y_val.extend(label)

    # patch test set
    for (img, gt) in zip(test, test_gt):
        patch, label = patches_supervised(img=img, gt=gt, size=patch_size)
        X_test.extend(patch)
        y_test.extend(label)
        
        
else: # unsupervised = self supervised pretraining
    
    criterion = torch.nn.MSELoss().to(device)

    for img in train:
    # I keep the coords so as to know which pixel I infer every time
        patch, coord_set = patches_unsupervised(img=img, size=patch_size)
        X_train.extend(patch)
        coords.extend(coord_set)

"""Dataloaders"""

## Dataset
train = HyRank(X=X_train, y=y_train, transform=available_transformations, 
               supervision=supervision)
val = HyRank(X=X_val, y=y_val, transform=False, supervision=supervision)
test = HyRank(X=X_test, y=y_test, transform=False, supervision=supervision)

# Dataloaders
from torch.utils.data import DataLoader
n = workers
trainloader = DataLoader(train, batch_size=batch, num_workers=n, shuffle=True)
valloader = DataLoader(val, batch_size=batch, num_workers=n, shuffle=False)
testloader = DataLoader(test, batch_size=batch, num_workers=n, shuffle=False)
 
"""Training"""

torch.cuda.empty_cache()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
ylimit = 1.5 if supervision else 0.003
trainNN(model=model, patience=patience, model_folder_name=selected_model, 
        batch=batch, patch_size=patch_size, max_epochs=max_epochs, 
        trainloader=trainloader, valloader=valloader, testloader=testloader, 
        device=device, optimizer=optimizer, criterion=criterion, lr=lr, 
        transform=available_transformations, save_to=save_to, y_limit=ylimit,
        classification=supervision, supervision=supervision,
        pre_model=pre_model, dropout=dropout)
