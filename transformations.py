import os
from scipy import ndarray
from scipy import ndimage
import skimage as sk
import random
from skimage import util

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-40, 40)
    return ndimage.rotate(input=image_array, angle=random_degree, axes=(1,2),
                          reshape=False)

def random_noise(image_array: ndarray):
    # add a bit of random noise to the image
    return sk.util.random_noise(image_array, mode="gaussian", mean=0, 
                                var=0.0001, clip=True)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, :, ::-1]

def vertical_flip(image_array: ndarray):
    # vertical flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1, :]

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
#    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'vertical_flip': vertical_flip,
#    'No': lambda x : x
}
