import os
import sys
current_dir_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.abspath(current_dir_path), './'))

import numpy as np
from tqdm import tqdm
import json

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

from data_preprocess import data_preprocess
from config_parser import config_parse
from data_preprocess import data_preprocess
from network_parser import create_network


def read_test_data(data_path, id_path):
    data = open(data_path).readlines()
    data = list(map(lambda line: line.strip(), data))

    ids = open(id_path).readlines()
    ids = np.array(list(map(lambda line: int(line.strip()), ids)))

    return data, ids


def get_features(image_data, batch, root_path, dataset, model_input_size, sess, input_node, output_node, is_training=None):
    image_features = []

    start = 0
    data_len = len(image_data)
    for end in tqdm(range(1,data_len+1)):
        if end % batch == 0 or end == data_len:
            batch_images = []
            for image_name in image_data[start:end]:
                image_path = os.path.join(root_path, dataset, image_name)
                image = data_preprocess(image_path, model_input_size)
                batch_images.append(image)
            batch_images = np.array(batch_images)
            if is_training is None:
                batch_features = sess.run(output_node, feed_dict={input_node: batch_images})
            else:
                batch_features = sess.run(output_node, feed_dict={input_node: batch_images, is_training: False})
            image_features.append(batch_features)

            start = end
    image_features = np.concatenate(image_features, axis=0)
    image_features_norm = np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-12
    l2_image_features = image_features / image_features_norm

    return l2_image_features


def get_result(query_features, query_labels, gallery_features, gallery_labels, top_k):
    query_len = len(query_features)

    pre_labels = []
    for i in tqdm(range(query_len)):
        query_feature = np.expand_dims(query_features[i], axis=0)
        cosine_similarity = np.sum(query_feature * gallery_features, axis=-1)
        distance_matrix = 1.0 - cosine_similarity
        sorted_inds = np.argsort(distance_matrix)
        res = gallery_labels[sorted_inds[:top_k[-1]]]
        pre_labels.append(res)

    pre_labels = np.array(pre_labels)
    pre_results = pre_labels == np.expand_dims(query_labels, axis=1)

    top_k_res = {}
    for k in top_k:
        recall_num = np.sum(np.max(pre_results[:,:k], axis=1)).item()
        top_k_res[k] = 1.0 * recall_num / query_len

    return top_k_res


def main():
    result_path =  'results/deepfashion2_retrieval/resnet50/1'
    config_path = os.path.join(result_path, 'config.yaml')
    config = config_parse(config_path)
    print('config:')
    print(json.dumps(config, indent=4))

    root_path = config['data']['root_path']
    dataset = config['data']['dataset']

    model_config = config['model']
    model_input_size = model_config['input_size']
    embedding_dim = model_config['embedding_dim']
    backbone_config = model_config['backbone']

    with tf.Graph().as_default():
        image_inputs = tf.placeholder(tf.float32, shape=(None, *model_input_size, 3))
        out, emb_out = create_network(image_inputs, False, backbone_config['name'], embedding_dim)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        sess.run(tf.global_variables_initializer())

        variables = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(variables)
        saver.restore(sess, os.path.join(result_path, 'best_checkpoint'))

        query_images, query_ids = read_test_data(**config['data']['test']['query'])
        query_features = get_features(query_images, 128, root_path, dataset, model_input_size, sess, image_inputs, emb_out)

        gallery_images, gallery_ids = read_test_data(**config['data']['test']['gallery'])
        gallery_features = get_features(gallery_images, 128, root_path, dataset, model_input_size, sess, image_inputs, emb_out)

        top_k = list(range(1,101))
        top_k_results = get_result(query_features, query_ids, gallery_features, gallery_ids, top_k)

        top_result = os.path.join(result_path, 'top_result.txt')
        with open(top_result, 'w') as f:
            for i in range(1,101):
                p = top_k_results[i]
                f.write('{}\t{:.4f}\n'.format(i, p))

        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            [emb_out.op.name])

        graph_io.write_graph(constant_graph, result_path, 'frozen_model.pb', as_text=False)


if __name__ == '__main__':
    main()
