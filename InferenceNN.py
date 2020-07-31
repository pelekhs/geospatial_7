"""Data loading"""

from import_images import *
# keep image dimensions for the outpur reconstruction
erato_dimensions = (erato.shape[1], erato.shape[2])
nefeli_dimensions = (nefeli.shape[1], nefeli.shape[2])
kirki_dimensions = (kirki.shape[1], kirki.shape[2]) 

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
#device = torch.device("cpu")

"""Dataset"""

from dataset import *

"""Training tools"""

from training import trainNN

"""Settings and hyperparameters"""

from config import *

"""Models"""

from models import *

pre_model = AE(supervision=False).float()
pre_model.load_state_dict(torch.load(pretrained_encoder))
pre_model.to(device)
pre_model.eval()

model = models["Classifier_supervised"].to(device)
model.load_state_dict(torch.load(clf_dict))
model.to(device)
model.eval() 

""" Patch Extraction """

# Keep the coords so as to know which pixel I infer every time
erato, erato_coords = patches_unsupervised(img=erato, size=patch_size,
                                           subsample=False)
nefeli, nefeli_coords = patches_unsupervised(img=nefeli, size=patch_size,
                                             subsample=False)
kirki, kirki_coords = patches_unsupervised(img=kirki, size=patch_size,
                                           subsample=False)

"""Dataloaders"""

from torch.utils.data import DataLoader

test1 = HyRank(X=erato, y=erato_coords, transform=False, supervision=True)
test2 = HyRank(X=nefeli, y=nefeli_coords, transform=False, supervision=True)
test3 = HyRank(X=kirki, y=kirki_coords, transform=False, supervision=True)


n = workers
testloader1 = DataLoader(test1, batch_size=512, num_workers=n, shuffle=False)
testloader2 = DataLoader(test2, batch_size=512, num_workers=n, shuffle=False)
testloader3 = DataLoader(test3, batch_size=512, num_workers=n, shuffle=False)

def get_predictions(logits):
    """ This function computes the vector of predictions given the 
        logit tensors """
    max_vals, y_pred = torch.max(logits, 1)
    return y_pred

""" Inference of 3 test images and inference map plots and saving"""

import tqdm
from PIL import Image
import matplotlib.pyplot as plt

with torch.no_grad():
    for testloader in [testloader1, testloader2, testloader3]:
        coords = []
        y_pred = []
        with tqdm.tqdm(total=len(testloader)) as pbar:
            for i, data in enumerate(testloader):
                # pass batches from the encoder
                inputs = pre_model.encode(data['tensors'].float().to(device))
                # then feed to the classifier and produce logits
                logits = model(inputs.float())
                # transform logits to class predictions
                batch_pred = get_predictions(logits)
                # append to the whole vector of predictions
                y_pred.extend(batch_pred.tolist())
                # also keep track of the predicted pixel coordinates
                coords.extend(data['labels'].tolist())
                pbar.update()

        if testloader == testloader1:
            width = erato_dimensions[0]
            height = erato_dimensions[1]
            erato_inference = np.asarray(y_pred).reshape((width, height))
            # save
            Image.fromarray(erato_inference.astype(np.uint8)) \
                .save("Erato_INF_NN.tif", format="tiff")
            # plot
            plt.imshow(erato_inference)
            plt.title("Erato infered classes")
            plt.show()
            
        elif testloader == testloader2:    
            width = nefeli_dimensions[0]
            height = nefeli_dimensions[1]
            nefeli_inference = np.asarray(y_pred).reshape((width, height))
            # save
            Image.fromarray(nefeli_inference.astype(np.uint8)) \
                .save("Nefeli_INF_NN.tif", format="tiff")
            plt.imshow(nefeli_inference)
            plt.title("Nefeli infered classes")
            plt.show()
        else:    
            width = kirki_dimensions[0]
            height = kirki_dimensions[1]
            kirki_inference = np.asarray(y_pred).reshape((width, height))
            # save
            Image.fromarray(kirki_inference.astype(np.uint8)) \
                .save("Kirki_INF_NN.tif", format="tiff")
            plt.imshow(kirki_inference)
            plt.title("Kirki infered classes")
            plt.show()