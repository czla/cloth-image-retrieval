import cv2
import numpy as np


def random_resolution(img, crop_size, min_ratio, max_ratio, prob_keep_aspect_ratio=0.5):
    min_crop_size = (int(crop_size[0] * min_ratio), int(crop_size[1] * min_ratio))
    max_crop_size = (int(crop_size[0] * max_ratio), int(crop_size[1] * max_ratio))

    if np.random.rand() < prob_keep_aspect_ratio:
        h, w, _ = img.shape
        r = np.random.rand()
        if h > w:
            h_new = min_crop_size[0] + r * (max_crop_size[0] - min_crop_size[0])
            w_new = w / h * h_new
        else:
            w_new = min_crop_size[1] + r * (max_crop_size[1] - min_crop_size[1])
            h_new = h / w * w_new
    else:
        r = np.random.rand()
        h_new = min_crop_size[0] + r * (max_crop_size[0] - min_crop_size[0])
        w_new = min_crop_size[1] + r * (max_crop_size[1] - min_crop_size[1])

    h_new = int(np.ceil(h_new))
    w_new = int(np.ceil(w_new))

    new_image = cv2.resize(img, (w_new, h_new))

    return new_image


if __name__ == '__main__':
    image_path = '/home/hxcai/Work/data/deepfashion2_retrieval/train/13/50/2/000668_item2_shop.jpg'
    image = cv2.imread(image_path)

    new_image = random_resolution(image, (256,256), 0.8, 1.8)

    cv2.imshow('image', image)
    cv2.imshow('new_image', new_image)
    cv2.waitKey(0)