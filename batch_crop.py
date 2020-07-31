import numpy as np

def crop_around_pixel(img, y, x, size=5):
    """Crops a hyperspe patch of certain size around a (y, x) pixel """
    startx = x - size//2
    starty = y - size//2
    return img[:, starty:starty+size, startx:startx+size]

def labeled_pixels(gt):
  """Locates labeled pixels in ground truth image"""
  return np.argwhere(gt > 0)[:, [1, 2]]

def pad_image(img, padding):
    """Pads initial large image according to the patch-size selected, so that
        patches do not exceed borders"""
    padded = np.zeros((img.shape[0], img.shape[1] + 2 * padding,
                      img.shape[2] + 2 * padding))
    padded[:, padding:-padding, padding:-padding] = img
    return padded

def patches_supervised(img, gt, size=5):
    """Creates tiles around every labeled pixel of image (supervised approach)
        will be used for supervised training only"""
    padding = size // 2
    img = pad_image(img, padding)
    coords = labeled_pixels(gt)
    patches = [crop_around_pixel(img, yx[0] + padding, yx[1] + padding, size)
              for yx in coords]
    # supervised approach ignores 0 labels of unclassified pixels so -1
    labels =  np.asarray([gt[:, yx[0], yx[1]] - 1 for yx in coords], dtype="uint8") \
              .flatten()
    return patches, labels


def patches_unsupervised(img, size=15, subsample=True):
    """Creates tiles around every pixel of image (unsupervised approach)
        will be used for inference and unsupervised pretraining"""
    padding = size // 2
    if subsample:
        coords = np.asarray(list(np.ndindex(img.shape[1], 
                                            img.shape[2])))[0:: size//2]
    else:
        coords = np.asarray(list(np.ndindex(img.shape[1], img.shape[2])))
    img = pad_image(img, padding)
    patches = [crop_around_pixel(img, yx[0] + padding, yx[1] + padding, size)
              for yx in coords]
    return patches, coords
