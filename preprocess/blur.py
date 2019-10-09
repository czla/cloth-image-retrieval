import cv2
import random


def random_blur(image, sizes):
    if random.random() < 0.5:
        s = random.choice(sizes)
        new_image = cv2.blur(image, ksize=(1,s))
    else:
        new_image = image

    return new_image
