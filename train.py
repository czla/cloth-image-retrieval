import os
import sys
current_dir_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.abspath(current_dir_path), './'))

import numpy as np
from tqdm import tqdm
import json

import tensorflow as tf

from config_parser import config_parse
from data_preprocess import data_preprocess
from network_parser import create_network, load_backbone
from loss_parser import *
from test import read_test_data, get_features, get_result


class DataLoader:
    def __init__(self, root_path, dataset, model_input_size, batch_p, batch_k, preprocess_config, data_config):
        self.root_path = root_path
        self.dataset = dataset
        self.model_input_size = model_input_size
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.preprocess_config = preprocess_config

        self.data = self.read_data(data_config['data'])
        self.data_len = len(self.data)

        if 'label' in data_config:
            self.label = self.read_data(data_config['label'])
            self.label = np.array(self.label, dtype=np.int32).squeeze()
        else:
            self.label = np.array(range(self.data_len))
        self.label_len = len(self.label)
        assert self.data_len == self.label_len

        self.data_inds = list(range(self.data_len))
        np.random.shuffle(self.data_inds)
        self.inds_im = np.zeros(self.data_len, dtype=np.int32)

        self.count = 0

    def read_data(self, data_path):
        data = list(open(data_path).readlines())
        data = list(map(lambda line: line.strip().split('\t'), data))

        return data

    def shuffle(self):
        self.count = 0
        np.random.shuffle(self.data_inds)

    def get_batch(self):
        image_data = []
        id_data = []
        label_data = []

        for _ in range(self.batch_p):
            data_idx = self.data_inds[self.count % self.data_len]
            each_p_data = self.data[data_idx]
            for _ in range(self.batch_k):
                each_p_k_data = each_p_data[self.inds_im[data_idx] % len(each_p_data)]
                image_path = os.path.join(self.root_path, self.dataset, each_p_k_data)
                image = data_preprocess(image_path, self.model_input_size, self.preprocess_config)
                image_data.append(image)
                id_data.append(data_idx)
                label_data.append(self.label[data_idx])

                self.inds_im[data_idx] += 1
                if self.inds_im[data_idx] == len(each_p_data):
                    self.inds_im[data_idx] = 0
                    np.random.shuffle(each_p_data)

            self.count += 1
            if self.count >= self.data_len:
                np.random.shuffle(self.data_inds)

        image_data = np.array(image_data)
        id_data = np.array(id_data)
        label_data = np.array(label_data)

        return image_data, id_data, label_data


def loss_average(count, pre_avg_loss_dict, new_loss_dict):
    def single_loss_average(ct, pvl, nl):
        nvl = 1.0 * (ct - 1) / ct * pvl + 1.0 * nl / ct
        return nvl

    new_avg_loss_dict = {}

    for key in pre_avg_loss_dict:
        pre_avg_loss = pre_avg_loss_dict[key]
        new_loss = new_loss_dict[key]
        if pre_avg_loss is None:
            new_avg_loss = new_loss
        else:
            if isinstance(pre_avg_loss, list):
                new_avg_loss = []
                for pvl, nl in zip(pre_avg_loss, new_loss):
                    nvl = single_loss_average(count, pvl, nl)
                    new_avg_loss.append(nvl)
            else:
                new_avg_loss = single_loss_average(count, pre_avg_loss, new_loss)
        new_avg_loss_dict[key] = new_avg_loss

    return new_avg_loss_dict


opt_dict = {
    'adam': tf.train.AdamOptimizer,
}


def main():
    config = config_parse('config.yaml')
    print('config:')
    print(json.dumps(config, indent=4))

    root_path = config['data']['root_path']
    dataset = config['data']['dataset']

    model_config = config['model']
    model_input_size = model_config['input_size']
    embedding_dim = model_config['embedding_dim']
    backbone_config = model_config['backbone']

    save_dir = os.path.join(config['save']['dir'], dataset, backbone_config['name'], str(config['save']['id']))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.system('cp config.yaml {}'.format(save_dir))
    else:
        raise Exception('save dir exists: {}'.format(save_dir))

    train_config = config['train']
    epoches = train_config['epoches']
    iters =train_config['iters']
    batch_p = train_config['batch_p']
    batch_k = train_config['batch_k']
    preprocess_config = train_config['preprocess']

    train_data = DataLoader(root_path, dataset, model_input_size, batch_p, batch_k, preprocess_config, config['data']['train'])
    query_images, query_ids = read_test_data(**config['data']['test']['query'])
    gallery_images, gallery_ids = read_test_data(**config['data']['test']['gallery'])

    with tf.Graph().as_default():
        image_inputs = tf.placeholder(tf.float32, shape=(None, *model_input_size, 3))
        is_training = tf.placeholder(dtype=tf.bool)
        id_inputs = tf.placeholder(tf.int32, shape=(None,))
        label_inputs = tf.placeholder(tf.int32, shape=(None,))

        out, emb_out = create_network(image_inputs, is_training, backbone_config['name'], embedding_dim)

        total_loss = 0.0
        total_loss_dict = {}
        for loss_name in train_config['loss']:
            if loss_name == 'dml':
                dml_loss = get_dml_loss(emb_out, id_inputs, train_config['loss'][loss_name])
                total_loss += dml_loss
                total_loss_dict[loss_name] = [dml_loss]
            elif loss_name == 'cls':
                cls_loss = get_cls_loss(emb_out, label_inputs, train_config['loss'][loss_name])
                total_loss += cls_loss
                total_loss_dict[loss_name] = [cls_loss]
            elif loss_name == 'horde':
                horde_loss = get_horde_loss(out, id_inputs, embedding_dim, train_config['loss']['dml'], train_config['loss'][loss_name])
                total_loss += tf.reduce_sum(horde_loss)
                total_loss_dict[loss_name] = horde_loss
            else:
                raise Exception('Unknown loss name: {}'.format(loss_name))
        total_loss_dict['total_loss'] = total_loss

        global_step = tf.Variable(0, name='global_step', trainable=False)

        lr_boundaries = [x*iters for x in train_config['lr']['boundaries']]
        lr_values = train_config['lr']['values']
        learning_rate = tf.train.piecewise_constant(x=global_step, boundaries=lr_boundaries, values= lr_values)

        optimizer = opt_dict[train_config['optimizer']](learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        sess.run(tf.global_variables_initializer())

        # load pretrain
        if backbone_config['pretrain']:
            load_backbone(sess, backbone_config)

        saver = tf.train.Saver()

        best_top1_recall = 0.0
        for epoch in range(epoches):
            train_data.shuffle()

            print('Epoch {}/{}'.format(epoch+1, epoches))

            print('train')
            count = 0
            loss_avg_dict = {key: None for key in total_loss_dict.keys()}
            pbar = tqdm(range(iters), dynamic_ncols=True)
            for _ in pbar:
                image_batch, id_batch, label_batch = train_data.get_batch()
                _, total_loss_dict_out, lr = sess.run([train_op, total_loss_dict, learning_rate], feed_dict={image_inputs: image_batch, id_inputs: id_batch, label_inputs: label_batch, is_training: True})

                count += 1
                loss_avg_dict = loss_average(count, loss_avg_dict, total_loss_dict_out)
                s = 'lr: {:.6f}, loss: {:.4f} ({})'.format(lr, loss_avg_dict['total_loss'], ', '.join(['{}: {}'.format(key, ','.join(['{:.4f}'.format(l) for l in loss_avg_dict[key]])) for key in loss_avg_dict if key != 'total_loss']))
                pbar.set_description(s)

            print('test')
            print('get query images feature')
            query_features = get_features(query_images, 128, root_path, dataset, model_input_size, sess, image_inputs, emb_out, is_training)
            print('get gallery images feature')
            gallery_features = get_features(gallery_images, 128, root_path, dataset, model_input_size, sess, image_inputs, emb_out, is_training)
            print('get results')
            top_k = [1,10,20,30,40,50]
            top_k_results = get_result(query_features, query_ids, gallery_features, gallery_ids, top_k)
            print('Recall@K')
            for k in top_k:
                print('Recall@{}: {:.4f}'.format(k, top_k_results[k]))

            top_1_result = top_k_results[1]
            if top_1_result > best_top1_recall:
                print('recall@1 increase ({:.3f} --> {:.3f}), save model'.format(best_top1_recall, top_1_result))
                saver.save(sess, '{}/best_checkpoint'.format(save_dir))
                with open('{}/best_result.txt'.format(save_dir), 'w') as f:
                    f.write('epoch: {}\n'.format(epoch+1))
                    f.write('Recall@K\n')
                    for k in top_k:
                        f.write('Recall@{}: {:.4f}\n'.format(k, top_k_results[k]))
                    best_top1_recall = top_1_result


if __name__ == '__main__':
    main()
