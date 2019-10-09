import cv2

from preprocess import crop, flip, distort, blur, normalize, erase, subsample, rotate, multiresolution
import numpy as np


def data_preprocess(image_path, dst_size, config=None):
    new_image = cv2.imread(image_path)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    if config is not None:
        if 'multiresolution' in config:
            new_image = multiresolution.random_resolution(new_image, crop_size=dst_size, **config['multiresolution'])
        if 'random_crop' in config and config['random_crop']:
            new_image = crop.random_crop(new_image, crop_size=dst_size)
        if 'random_resized_crop' in config:
            new_image = crop.random_resized_crop(new_image, size=dst_size, **config['random_resized_crop'])
        if 'random_subsample':
            new_image = subsample.random_subsample(new_image, **config['random_subsample'])
        if 'horizontal_flip' in config:
            new_image = flip.random_horizontal_flip(new_image, config['horizontal_flip'])
        if 'vertical_flip' in config:
            new_image = flip.random_vertical_flip(new_image, config['vertical_flip'])
        # new_image = rotate.random_rotate(new_image, angles=(0,1,2,3))
        # new_image = resize.random_resize(new_image, scale=0.1)
        # new_image = blur.random_blur(new_image, sizes=[3,5,7])
        # new_image = distort.random_distort(new_image, hue=0.0, sat=1.5, val=1.5)
        # new_image = erase.random_erasing(new_image, probability = 0.5, sl = 0.02, sh = 0.09, r1 = 0.3, value=127.5)

    new_image = cv2.resize(new_image, (dst_size[1], dst_size[0]))
    new_image = new_image.astype(np.float32)
    new_image /= 127.5
    new_image -= 1.0

    return new_image


if __name__ == '__main__':
    from config_parser import config_parse

    config = config_parse('config.yaml')

    image_path = '/home/hxcai/Work/data/deepfashion2_retrieval/train/13/50/2/000668_item2_shop.jpg'
    image = data_preprocess(image_path, config['model']['input_size'], config['train']['preprocess'])

    print(image)
