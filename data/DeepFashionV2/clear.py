import os
from tqdm import tqdm

if __name__ == '__main__':
    root_path = '/home/hxcai/Work/data/deepfashion2_retrieval'
    for data_set in tqdm(os.listdir(root_path)):
        for category_id in os.listdir(os.path.join(root_path, data_set)):
            for pair_id in os.listdir(os.path.join(root_path, data_set, category_id)):
                for style in os.listdir(os.path.join(root_path, data_set, category_id, pair_id)):
                    for name in os.listdir(os.path.join(root_path, data_set, category_id, pair_id, style)):
                        image_file = os.path.join(root_path, data_set, category_id, pair_id, style, name)
                        size = os.path.getsize(image_file)
                        if size == 0:
                            print(image_file)