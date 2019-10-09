import cv2
import random
import math
import numpy as np


def random_resized_crop(img, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    def get_params(img, scale, ratio):
        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
                w, h = h, w

            if h <= img.shape[0] and w <= img.shape[1]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w
        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    if random.random() < 0.5:
        i, j, h, w = get_params(img, scale, ratio)
        new_image = img[i:i+h,j:j+w]
    else:
        new_image = img

    new_image = cv2.resize(new_image, (size[1], size[0]))

    return new_image


def random_crop(img, crop_size):
    h, w, _ = img.shape
    if h == crop_size[0] and w == crop_size[1]:
        return img

    h_off_range = h - crop_size[0]
    w_off_range = w - crop_size[1]

    if h_off_range > 0:
        h_off = np.random.randint(0, h_off_range, 1, dtype=np.int32)[0]
        h_m = crop_size[0]
    else:
        h_off = 0
        h_m = h

    if w_off_range > 0:
        w_off = np.random.randint(0, w_off_range, 1, dtype=np.int32)[0]
        w_m = crop_size[1]
    else:
        w_off = 0
        w_m = w

    new_image = img[h_off:h_off+h_m, w_off:w_off+w_m, :]
    new_image = cv2.resize(new_image, (crop_size[1], crop_size[0]))

    return new_image


if __name__ == '__main__':
    import multiresolution

    image_path = '/home/hxcai/Work/data/deepfashion2_retrieval/train/12/72/1/000928_item1_user.jpg'
    image = cv2.imread(image_path)

    new_resolution_image = multiresolution.random_resolution(image, (192,192), 0.8, 1.2)
    new_crop_image = random_crop(new_resolution_image, (192,192))
    new_resized_crop_image = random_resized_crop(new_resolution_image, (192, 192), (0.8, 1.0), (0.6, 1.66))

    cv2.imshow('image', image)
    cv2.imshow('new_resolution_image', new_resolution_image)
    cv2.imshow('new_crop_image', new_crop_image)
    cv2.imshow('new_resized_crop_image', new_resized_crop_image)
    cv2.waitKey(0)

    # new_crop_image = random_resized_crop(image, (256,256), scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.))
    # cv2.imshow('image', image)
    # cv2.imshow('new_crop_image', new_crop_image)
    # cv2.waitKey(0)