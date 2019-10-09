import cv2
import numpy as np
import random
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0., b=1.):
    return np.random.rand()*(b-a) + a


def random_distort(img, hue=0.0, sat=1.5, val=1.5):
    if random.random() < 0.5:
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(img/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) * 255.
    else:
        image_data = img

    return image_data


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    image = cv2.imread('/home/hxcai/Pictures/data/20190715/02/00033/42/20190715_020003342_09043801.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_out = random_distort(image)
    print(image_out)
    plt.imshow(image_out.astype(np.int32))
    plt.show()
