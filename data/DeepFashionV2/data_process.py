import os
import json
import cv2
from tqdm import tqdm


category_names = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt',
                  'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']
category_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


if __name__ == '__main__':
    root_path = '/home/hxcai/Work/data/deepfashion2'
    res_path = '/home/hxcai/Work/data/deepfashion2_retrieval'

    data_set = 'train'
    annos_dir = os.path.join(root_path, data_set, 'annos')
    image_dir = os.path.join(root_path, data_set, 'image')
    for name in tqdm(os.listdir(image_dir)):
        image_name, format = name.split('.')
        json_name = '%s.json' % image_name
        json_file = os.path.join(annos_dir, json_name)
        if os.path.exists(json_file):
            info = json.load(open(json_file))
            if 'source' in info and 'pair_id' in info:
                image_file = os.path.join(image_dir, name)
                image = cv2.imread(image_file)
                source = info['source']
                pair_id = info['pair_id']
                for key in info.keys():
                    if 'item' in key:
                        item = info[key]
                        style = item['style']
                        category_id = item['category_id']
                        category_name = item['category_name']
                        x1, y1, x2, y2 = item['bounding_box']
                        cloth = image[y1:y2, x1:x2, :]
                        save_dir = os.path.join(res_path, data_set, str(category_id), str(pair_id), str(style))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_name = '%s_%s_%s.jpg' % (image_name, key, source)
                        save_path = os.path.join(save_dir, save_name)
                        if not os.path.exists(save_path):
                            cv2.imwrite(save_path, cloth)
