# =============================================================================
# API
# =============================================================================
import os

# set true so as to train classifiers on all of Dioni and Loukia (final stage)
train_all = False

# lr and regularization
lr = 1e-4
regularizer = 0
dropout = 0.0

# Number of processors
workers = 0

# set patch size
patch_size = 11

# set batch size
batch = 64

# supervision -> True or False
supervision = True

# Transformations
transforms = False

# select model here
selected_model = "AE_new"

# Path to state dict of pretrained autoencoder else None
pretrained_encoder = "../results/AE_new_unsupervised/11_False_cloud/state_dict" 

# Location to save models
save_to = str(os.path.join(os.pardir, "results"))

# Patience to stop training after best model
patience = 20

max_epochs = 50

# =============================================================================
# End of API
# =============================================================================

# Do not change below

import os
from models import *
import torch

supervised = "supervised" if supervision else "unsupervised"
models = {"old_AE_" + supervised: old_AE(supervision=supervision),
          "AE_new_" + supervised: AE(supervision=supervision),
          "AE3D_" + supervised: AE3D(supervision=supervision),
          "Classifier_supervised": Classifier(dropout=dropout)}
pre_model=None

# transforms
from transformations import *
available_transformations = available_transformations if transforms else False

# model
selected_model = selected_model + "_" + supervised

if pretrained_encoder:
    supervision = True
    selected_model = "Classifier_supervised"
    
    pre_model = AE(supervision=False).float()
    pre_model.load_state_dict(torch.load(pretrained_encoder))
    pre_model.eval()
    
    print("\nApplying fine tuning using pretrained encoder outputs\n")

