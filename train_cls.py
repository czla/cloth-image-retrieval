import os
import sys
current_dir_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.abspath(current_dir_path), './'))

import tensorflow as tf
import cv2
import numpy as np

from network.c2s_attention import googlenet, load_weights
from network import resnet, resnet_spatial, resnet_hoa
from loss import batch_hard_loss, margin_sample_mining_loss, margin_based_loss, horde_loss, ms_loss, adv_loss, ep_loss, softmax_loss
from preprocess import crop, flip, distort, blur, normalize, erase, resize, rotate


class DataLoader:
    def __init__(self, data_path, size, batch_p, batch_k, is_training, root_path, label_path):
        self.size = size
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.root_path = root_path
        self.is_training = is_training
        self.label_path = label_path

        self.data = self.read_data(data_path)
        self.data_len = len(self.data)
        self.batch = 0

        self.label_data = self.read_data(self.label_path)
        self.label_data_len = len(self.label_data)
        assert self.data_len == self.label_data_len, "label data length doesn't match!"

        self.indexs = list(range(self.data_len))

        self.random_shuffle()

    def read_data(self, data_path):
        data = list(open(data_path).readlines())
        data = list(map(lambda line: line.strip().split('\t'), data))

        return data

    def random_shuffle(self):
        np.random.shuffle(self.indexs)

    def data_preprocess(self, image):
        if self.is_training:
            new_image = flip.random_horizontal_flip(image)
            new_image = flip.random_vertical_flip(new_image)
            new_image = crop.random_resized_crop(new_image, self.size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.))
            # new_image = rotate.random_rotate(new_image, angles=(0,1,2,3))
            # new_image = resize.random_resize(new_image, scale=0.1)
            # new_image = blur.random_blur(new_image, sizes=[3,5,7])
            # new_image = distort.random_distort(new_image, hue=0.0, sat=1.5, val=1.5)
            # new_image = erase.random_erasing(new_image, probability = 0.5, sl = 0.02, sh = 0.09, r1 = 0.3, value=127.5)
        else:
            new_image = cv2.resize(image, self.size[::-1])

        mean = [0.59763736, 0.5482761, 0.5331988]
        std = [0.30011338, 0.30333182, 0.2868211]
        new_image = new_image / 255.
        new_image = (new_image - mean) / std

        return new_image

    def get_batch(self):
        image_data = []
        label_data = []

        for p in range(self.batch_p):
            index = self.indexs[(self.batch+p) % self.data_len]
            each_p_data = self.data[index]
            all_data = each_p_data.copy()
            while len(all_data) < self.batch_k:
                all_data += all_data
            np.random.shuffle(all_data)
            for each_p_k_data in all_data[0:self.batch_k]:
                image = cv2.imread(os.path.join(self.root_path, each_p_k_data))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.data_preprocess(image)
                image_data.append(image)
                label_data.append(self.label_data[index])

        image_data = np.array(image_data, dtype=np.float32)
        label_data = np.array(label_data, dtype=np.int32).squeeze()

        self.batch += self.batch_p
        if self.batch > self.data_len:
            self.batch = 0
            self.random_shuffle()

        return image_data, label_data


def main():
    model_input_size = (224, 224)
    batch_p = 32
    batch_k = 3
    loss_func = ep_loss
    loss_name = loss_func.__name__
    metric = 'euclidean'
    margin = 0.2
    beta = 1.2
    orders = 2
    embedding_dim = 128
    model_name = 'resnet50_%d' % embedding_dim
    train_iters = 130000
    boundaries = [45000, 90000]
    values = [1e-4, 1e-5, 1e-6]
    num_warmup_steps = 1000
    test_interval = 1000
    val_iters = 200
    test_iters = 100
    display_iters = 10
    save_iters = 5000

    root_path = '/home/zlchen/dataset/deepfashion2_retrieval'
    train_label_path = 'data/DeepFashionV2/train_label.txt'
    val_label_path = 'data/DeepFashionV2/val_label.txt'
    train_data = DataLoader('data/DeepFashionV2/train_set.txt', model_input_size, batch_p, batch_k, True, root_path, train_label_path)
    image_batch, label_batch = train_data.get_batch()

    val_data = DataLoader('data/DeepFashionV2/validation_set.txt', model_input_size, batch_p, batch_k, False, root_path, val_label_path)
    # image_batch, label_batch = train_data.get_batch()
    # test_data = DataLoader('data/DeepFashion/test.txt', model_input_size, batch_p, batch_k, False)

    log_file = open('logs/log_{}_{}.txt'.format(model_name, loss_name), 'w')

    with tf.Graph().as_default():
        image_inputs = tf.placeholder(tf.float32, shape=(None, *model_input_size, 3))
        label_inputs = tf.placeholder(tf.int32, shape=(None,))
        is_training = tf.placeholder(dtype=tf.bool)

        # fc_out = resnet.resnet50(image_inputs, is_training, embedding_dim)
        # fc_out = resnet.resnet50(image_inputs, is_training, embedding_dim, orders=orders)
        out, fc_out = resnet.resnet50(image_inputs, is_training, embedding_dim, before_pool=True)

        # dml_loss = loss_func.loss(fc_out, label_inputs, margin=margin, metric=metric)
        # dml_loss = loss_func.loss(fc_out, label_inputs, lamb=margin, ms_mining=True)
        dml_loss = loss_func.loss(fc_out, label_inputs)
        # ho_loss = adv_loss.loss(fc_outs, is_training)
        ho_loss = softmax_loss.loss(out, label_inputs)
        loss = dml_loss + ho_loss

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(x=global_step, boundaries=boundaries, values=values)

        # warm  up
        # global_steps_int = tf.cast(global_step, tf.int32)
        # warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        #
        # global_steps_float = tf.cast(global_steps_int, tf.float32)
        # warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        #
        # warmup_percent_done = global_steps_float / warmup_steps_float
        # warmup_learning_rate = values[0] * warmup_percent_done
        #
        # is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        # learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        # optimizer
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        sess.run(tf.global_variables_initializer())

        print('model: {}'.format(model_name))
        print('embedding_dim: {}'.format(embedding_dim))
        print('loss name: {}'.format(loss_name))
        print('metric: {}'.format(metric))
        print('margin: {}'.format(margin))

        # warn up
        print('warm up')
        image_batch = np.random.randn(batch_p*batch_k, *model_input_size, 3)
        label_batch = np.random.randint(0, batch_p, (batch_p*batch_k,))
        sess.run(loss, feed_dict={image_inputs: image_batch, label_inputs: label_batch, is_training: False})

        # load pretrain
        print('load pretrain model')
        # load_weights(sess, 'weights/pretrain/c2s_attention/pretrain_c2s_attention.npy')
        restore_variables = [var for var in tf.trainable_variables() if 'resnet_v2_50' in var.name]
        restore_saver = tf.train.Saver(var_list=restore_variables)
        restore_saver.restore(sess=sess, save_path='weights/pretrain/resnet50/resnet_v2_50.ckpt')

        saver = tf.train.Saver()

        print('training begein')
        for train_iter in range(train_iters+1):
            if train_iter != 0 and train_iter % test_interval == 0:
                # val
                val_avg_loss = 0
                for _ in range(val_iters):
                    image_batch, label_batch = val_data.get_batch()
                    loss_out = sess.run(dml_loss, feed_dict={image_inputs: image_batch, label_inputs: label_batch, is_training: False})
                    val_avg_loss += loss_out
                val_avg_loss /= val_iters

                # # test
                # test_avg_loss = 0
                # for _ in range(test_iters):
                #     image_batch, label_batch = test_data.get_batch()
                #     loss_out = sess.run(loss, feed_dict={image_inputs: image_batch, label_inputs: label_batch, is_training: False})
                #     test_avg_loss += loss_out
                # test_avg_loss /= test_iters

                # print('train iter %d, val loss: %0.3f, test loss: %0.3f' % (train_iter, val_avg_loss, test_avg_loss), file=log_file, flush=True)
                print('train iter %d, val loss: %0.3f' % (train_iter, val_avg_loss), file=log_file, flush=True)

            if train_iter != 0 and train_iter % save_iters == 0:
                print('save model')
                saver.save(sess, 'weights/{}_{}/checkpoint'.format(model_name, loss_name), global_step=global_step)

            image_batch, label_batch = train_data.get_batch()
            if train_iter != 0 and train_iter % display_iters == 0:
                _, loss_out, dml_loss_out, ho_loss_out, train_lr = sess.run([train_op, loss, dml_loss, ho_loss, learning_rate],  feed_dict={image_inputs: image_batch, label_inputs: label_batch, is_training: True})
                print('train iter %d, lr: %s, loss: %0.3f, dml_loss: %0.3f, ho_loss: %0.3f' % (train_iter, train_lr, loss_out, dml_loss_out, ho_loss_out), file=log_file, flush=True)
            else:
                sess.run([train_op, loss], feed_dict={image_inputs: image_batch, label_inputs: label_batch, is_training: True})

    log_file.close()


if __name__ == '__main__':
    main()
