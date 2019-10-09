import random
import math


def random_erasing(image, probability = 0.5, sl = 0.02, sh = 0.09, r1 = 0.3, value=127.5):
    if random.uniform(0, 1) > probability:
        return image

    image_h, image_w, _ = image.shape
    for attempt in range(100):
        area = image_h * image_w

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < image_w and h < image_h:
            i = random.randint(0, image_h - h)
            j = random.randint(0, image_w - w)
            image[i:i + h, j:j + w] = value

            return image

    return image


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    import numpy as np

    image = cv2.imread('/home/hxcai/Pictures/test/1563433082069_1_13_1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    image = random_erasing(image)

    plt.figure(1)
    plt.imshow(image.astype(np.int32))
    plt.show()