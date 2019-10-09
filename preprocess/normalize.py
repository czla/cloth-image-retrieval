import numpy as np


def normalize(img, mean=(104.2657, 120.8229, 119.6200)):
    image_mean = np.array(mean)
    new_image = img - image_mean

    return new_image
