import os
from models import *
import torch

# =============================================================================
# API
# =============================================================================

# lr and regularization
lr = 1e-3
regularizer = 0
dropout = 0.0

# Number of processors
workers = 0

# set patch size
patch_size = 11

# set batch size
batch = 8

# supervision -> True or False
supervision = False

# Transformations
transforms = False

# select model here
selected_model = "AE_new"


# load pretrained encoder here -> torch_load or None
#pretrained_encoder = torch.load("../results/AE_new_unsupervised/100_epochs_div2/checkpoint.pt")
pretrained_encoder = None#"../results/AE_new_unsupervised/11_20200726-133027/state_dict" 

# Location to save models
save_to = str(os.path.join(os.pardir, "results"))

# Patience to stop training after best model
patience = 20

max_epochs = 20





# =============================================================================
#  Do not change
# =============================================================================
supervised = "supervised" if supervision else "unsupervised"
models = {"old_AE_" + supervised: old_AE(supervision=supervision),
          "AE_new_" + supervised: AE(supervision=supervision),
          "AE3D_" + supervised: AE3D(supervision=supervision),
          "Classifier_"+ "supervised": Classifier(dropout=dropout)}
pretrained_encoder=None
pre_model=None

# transforms
from transformations import *
available_transformations = available_transformations if transforms else False

# model
selected_model = selected_model + "_" + supervised

if pretrained_encoder != None:
    supervision = True
    selected_model = "Classifier_supervised"
    
    pre_model = AE(supervision=False).float()
    pre_model.load_state_dict(torch.load(pretrained_encoder))
    pre_model.eval()
    
    print("Applying fine tuning using pretrained encoder outputs")

