import cv2
import random


def random_vertical_flip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, flipCode=1)
    return img


def random_horizontal_flip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, flipCode=0)
    return img