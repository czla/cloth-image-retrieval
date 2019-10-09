import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops, nn_ops
import argparse
import numpy as np
import cv2
import os


def parse_args():
    """
    args for resnet visualization.
    """
    parser = argparse.ArgumentParser(description='visualize resnet heatmap')
    parser.add_argument('--img_path', default='/home/zlchen/dataset/deepfashion2_retrieval/train/1/1/1/000001_item1_user.jpg')
    parser.add_argument('--model_path', type=str, default='resnet50_128_loss.batch_hard_loss_euclidean_0.2')
    parser.add_argument('--root_path', default='/home/zlchen/scripts/internship/tf-cloth_image_retrieval')
    parser.add_argument('--query_id', default='')
    parser.add_argument('--gallery_id', default='1')
    parser.add_argument('--correct', default='False')
    args = parser.parse_args()

    return args

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if stride == 1:
        return layers_lib.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = array_ops.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

        return layers_lib.conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)


def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1):
    with tf.variable_scope('bottleneck_v2'):
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = layers.batch_norm(inputs, activation_fn=nn_ops.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = layers_lib.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')

        residual = layers_lib.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = layers_lib.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

    output = shortcut + residual

    return output


def resnet_v2_block(inputs, base_depth, num_units, stride, rate=1, scope=None):
    out = inputs
    with tf.variable_scope(scope):
        for i in range(num_units-1):
            with tf.variable_scope('unit_%d' % (i+1)):
                out = bottleneck(inputs=out, depth=base_depth*4, depth_bottleneck=base_depth, stride=1, rate=rate)
        with tf.variable_scope('unit_%d' % num_units):
            out = bottleneck(inputs=out, depth=base_depth*4, depth_bottleneck=base_depth, stride=stride, rate=rate)

    return out


def resnet50(image_input, is_training, embedding_dim, scope='resnet_v2_50', before_pool=False):
    with tf.variable_scope(scope):
        with arg_scope([layers.batch_norm], is_training=is_training, scale=True):
            with arg_scope([layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
                out = conv2d_same(inputs=image_input, num_outputs=64, kernel_size=7, stride=2, scope='conv1')
            out = layers.max_pool2d(out, [3, 3], stride=2, scope='pool1')
            with arg_scope([layers_lib.conv2d], activation_fn=nn_ops.relu, normalizer_fn=layers.batch_norm):
                out = resnet_v2_block(inputs=out, base_depth=64, num_units=3, stride=2, scope='block1')
                out = resnet_v2_block(inputs=out, base_depth=128, num_units=4, stride=2, scope='block2')
                out = resnet_v2_block(inputs=out, base_depth=256, num_units=6, stride=2, scope='block3')
                out = resnet_v2_block(inputs=out, base_depth=512, num_units=3, stride=1, scope='block4')
            out = layers.batch_norm(out, activation_fn=nn_ops.relu, scope='postnorm')
            avg_out = tf.reduce_mean(out, axis=[1,2], name='pool5')

    fc_out = tf.layers.dense(inputs=avg_out, units=embedding_dim)

    if before_pool:
        return out, fc_out
    else:
        return fc_out

def normalize(data):
    x_max = np.max(data)
    x_min = np.min(data)
    return (data - x_min)/(x_max - x_min)

def getActivatons(feature, shape):
    w, h, _ = shape
    feature = np.mean(feature, axis=3)
    feature = np.squeeze(feature)
    feature = normalize(feature)
    feature = cv2.resize(feature, (h, w), interpolation=cv2.INTER_LINEAR)
    return feature


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    img_path = args.img_path
    root_path = args.root_path
    query_id = args.query_id
    gallery_id = args.gallery_id
    correct = args.correct
    res_path = os.path.join(root_path, 'results', model_path, 'heatmap', query_id)
    with tf.Graph().as_default():
        # image_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        # is_training = tf.placeholder(dtype=tf.bool)
        # output = resnet50(image_input, is_training, 128)

        # tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        # sess = tf.Session(config=tf_config)
        #
        # sess.run(tf.global_variables_initializer())
        #
        # for var in tf.all_variables():
        #     print(var.name)
        #
        # restore_variables = [var for var in tf.trainable_variables() if 'dense' not in var.name]
        # saver = tf.train.Saver(var_list=restore_variables)
        # saver.restore(sess=sess, save_path='../weights/pretrain/resnet50/resnet_v2_50.ckpt')


        # visualize a heatmap of a test image
        imageToUse = cv2.imread(img_path)

        image = cv2.resize(imageToUse, (224, 224), interpolation=cv2.INTER_NEAREST)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        # cv2.imshow('ori', imageToUse)
        output, _ = resnet50(image, is_training=False, embedding_dim=128, before_pool=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        sess.run(tf.global_variables_initializer())

        variables = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(variables)
        saver.restore(sess, '../weights/{}/checkpoint-110000'.format(model_path))

        feature = output.eval(session=sess)

        heatmap = getActivatons(feature, imageToUse.shape)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        heatmap = np.uint8(255 * heatmap)
        res = np.hstack([imageToUse, heatmap])

        # add label
        if gallery_id != '0':
            cv2.putText(res, correct, (imageToUse.shape[1], 35), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 3)

        # cv2.imshow('img', res)
        # cv2.waitKey(0)
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        filename = gallery_id + '_' + img_path.split('/')[-1]
        cv2.imwrite(os.path.join(res_path, filename), res)

    # print(output)