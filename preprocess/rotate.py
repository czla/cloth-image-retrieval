import cv2
import numpy as np
import random


def random_rotate(image, angles=(0,1,2,3)):
    new_image = image.copy()

    angle = random.choice(angles)

    for i in range(angle):
        new_image = np.rot90(new_image)

    return new_image


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    image = cv2.imread('/home/hxcai/Pictures/data/20190715/02/00033/42/20190715_020003342_09043801.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_out = random_rotate(image)
    print(image_out)
    plt.imshow(image_out.astype(np.int32))
    plt.show()