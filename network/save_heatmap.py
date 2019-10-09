import os
import random


def main():
    root_path = '/home/zlchen/scripts/internship/tf-cloth_image_retrieval'
    dataset_path = '/home/zlchen/dataset/deepfashion2_retrieval'
    result_file = os.path.join(root_path, 'results/resnet50_128_loss.batch_hard_loss_euclidean_0.2/result.txt')

    if not os.path.exists(result_file):
        raise FileNotFoundError('result file not found!')

    result_data = open(result_file).readlines()
    result_data = list(map(lambda line: line.strip().split('\t'), result_data))
    nums = 1
    query_id = 1
    correct = False


    for line in result_data[80:100]:

        # r = []
        _, consumer_category, consumer_pair_id, consumer_style, _ = line[0].split('/')
        for gallery_id, shop_name in enumerate(line[:6]):

            img_path = os.path.join(dataset_path, shop_name)

            print('##images {}\tSave heatmap for image: {}'.format(nums, img_path))
            # print('python resnet_visualization.py --img_path={}'.format(img_path))

            nums += 1
            _, shop_category, shop_pair_id, shop_style, _ = shop_name.split('/')
            if consumer_category == shop_category and consumer_pair_id == shop_pair_id and consumer_style == shop_style:
                correct = 'True'
            else:
                correct = 'False'

            os.system('python -W ignore resnet_visualization.py --img_path={} --query_id={} \
            --gallery_id={} --correct={}'.format(img_path, str(query_id), str(gallery_id), correct))
        query_id += 1

if __name__ == '__main__':
    main()