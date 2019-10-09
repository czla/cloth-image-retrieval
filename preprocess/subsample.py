import cv2
import numpy as np


def rand(a=0., b=1.):
    return np.random.rand()*(b-a) + a


def random_subsample(img, scale=0.1):
    h, w, _ = img.shape
    if np.random.rand() < 0.5:
        ratio = rand(scale, 1.0)
        nh, nw = int(h*ratio), int(w*ratio)
        new_image = cv2.resize(img, (nw, nh))
        new_image = cv2.resize(new_image, (w, h))
    else:
        new_image = img

    return new_image


if __name__ == '__main__':
    image = cv2.imread('/home/hxcai/Work/data/deepfashion2_retrieval/train/12/72/1/000928_item1_user.jpg')
    image_out = random_subsample(image)

    cv2.imshow('a', image)
    cv2.imshow('out', image_out)
    cv2.waitKey(0)
