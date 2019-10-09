import cv2
import numpy as np
import random


def rand(a=0., b=1.):
    return np.random.rand()*(b-a) + a


def random_resize(img, scale=0.1):
    h, w, _ = img.shape
    if random.random() < 0.5:
        ratio = rand(scale, 1.0)
        nh, nw = int(h*ratio), int(w*ratio)
        new_image = cv2.resize(img, (nw, nh))
        new_image = cv2.resize(new_image, (w, h))
    else:
        new_image = img

    return new_image


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    image = cv2.imread('/home/hxcai/Pictures/data/20190715/02/00033/42/20190715_020003342_09043801.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_out = random_resize(image)
    print(image_out)
    plt.imshow(image_out.astype(np.int32))
    plt.show()