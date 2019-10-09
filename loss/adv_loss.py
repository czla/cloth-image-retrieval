import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops, nn_ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops


def resnet_arg_scope(weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True, is_training=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
        'is_training': is_training
    }

    with arg_scope([layers_lib.conv2d],
                   weights_regularizer=regularizers.l2_regularizer(weight_decay),
                   weights_initializer=initializers.variance_scaling_initializer(),
                   activation_fn=nn_ops.relu,
                   normalizer_fn=layers.batch_norm,
                   normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params):
            with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def adv_encode(inputs, is_training, reuse=False):
    with tf.variable_scope('loss', reuse=reuse):
        with arg_scope(resnet_arg_scope(is_training=is_training)):
            out = tf.nn.l2_normalize(inputs, axis=1)
            out = layers_lib.fully_connected(out, 128, activation_fn=None, biases_initializer=None)
            out = tf.stop_gradient(2 * out) - out
            out = layers.batch_norm(out, activation_fn=None)
            out = tf.nn.relu(out)
            out = layers_lib.fully_connected(out, 128, activation_fn=None)
            out = layers.batch_norm(out, activation_fn=None)
            out = tf.nn.l2_normalize(out, axis=1)

    return out


def loss(outs, is_training):
    orders = len(outs)
    flag = False

    loss_out = 0
    for i in range(orders):
        for j in range(orders):
            if i < j:
                out_1 = adv_encode(outs[i], is_training, reuse=flag)
                flag = True
                out_2 = adv_encode(outs[j], is_training, reuse=flag)
                diff = out_1 - out_2
                diff_norm = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1)+1e-12)
                loss_out += tf.reduce_mean(diff_norm)

    loss_out = loss_out * 2 / ((orders-1) * orders)

    return loss_out


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = [tf.random_uniform((3,256)) for i in range(3)]
        is_training = False

        loss = loss(inputs, is_training)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for var in tf.global_variables():
            print(var.name)

        loss_output = sess.run(loss)
        print(loss_output)