import rasterio
import os
import numpy as np

#normalisation function for each band separately
normalise = lambda x : x / np.max(x)

# Dioni
# read
dioni = rasterio.open(os.path.join(os.pardir, 'data', 'HyRANK_satellite', 'TrainingSet', 'Dioni.tif')).read()
#normalise
dioni = np.array(list(map(normalise, [dioni[i, :, :] for i in range(dioni.shape[0])])))

# Dioni GT
dioni_gt = rasterio.open(os.path.join(os.pardir, 'data', 'HyRANK_satellite', 'TrainingSet', 'Dioni_GT.tif')).read()

# Loukia
loukia = rasterio.open(os.path.join(os.pardir, 'data', 'HyRANK_satellite', 'TrainingSet', 'Loukia.tif')).read()
loukia = np.array(list(map(normalise, [loukia[i, :, :] for i in range(loukia.shape[0])])))

# Loukia GT
loukia_gt = rasterio.open(os.path.join(os.pardir, 'data', 'HyRANK_satellite', 'TrainingSet', 'Loukia_GT.tif')).read()

# Erato
erato = rasterio.open(os.path.join(os.pardir, 'data', 'HyRANK_satellite', 'TestSet', 'Erato.tif')).read()
erato = np.array(list(map(normalise, [erato[i, :, :] for i in range(erato.shape[0])])))

#  Kirki
kirki = rasterio.open(os.path.join(os.pardir, 'data', 'HyRANK_satellite', 'TestSet', 'Kirki.tif')).read()
kirki = np.array(list(map(normalise, [kirki[i, :, :] for i in range(kirki.shape[0])])))

# Nefeli
nefeli = rasterio.open(os.path.join(os.pardir, 'data', 'HyRANK_satellite', 'TestSet', 'Nefeli.tif')).read()
nefeli = np.array(list(map(normalise, [nefeli[i, :, :] for i in range(nefeli.shape[0])])))
