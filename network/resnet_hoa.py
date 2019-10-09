import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops, nn_ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops


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


def resnet50_part1(image_input, is_training, scope='resnet_v2_50'):
    with tf.variable_scope(scope):
        with arg_scope(resnet_arg_scope(is_training=is_training)):
            with arg_scope([layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
                out = conv2d_same(inputs=image_input, num_outputs=64, kernel_size=7, stride=2, scope='conv1')
            out = layers.max_pool2d(out, [3, 3], stride=2, scope='pool1')
            out = resnet_v2_block(inputs=out, base_depth=64, num_units=3, stride=2, scope='block1')
            out = resnet_v2_block(inputs=out, base_depth=128, num_units=4, stride=2, scope='block2')

    return out


def resnet50_part2(inputs, is_training, embedding_dim, scope='resnet_v2_50', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope(resnet_arg_scope(is_training=is_training)):
            out = resnet_v2_block(inputs=inputs, base_depth=256, num_units=6, stride=2, scope='block3')
            out = resnet_v2_block(inputs=out, base_depth=512, num_units=3, stride=1, scope='block4')
            out = layers.batch_norm(out, activation_fn=nn_ops.relu, scope='postnorm')
            avg_out = tf.reduce_mean(out, axis=[1, 2], name='pool5')

    with tf.variable_scope('fc', reuse=reuse):
        with arg_scope(resnet_arg_scope(is_training=is_training)):
            fc_out = layers_lib.fully_connected(inputs=avg_out, num_outputs=embedding_dim, activation_fn=None)
            fc_out = layers.batch_norm(fc_out, activation_fn=None, scope='out_norm')

    return fc_out


def high_order_module(inputs, order, scope):
    with tf.variable_scope(scope):
        in_channels = inputs.shape[3]
        inter_channels = in_channels // 8 * 2

        attention = 0
        with arg_scope([layers_lib.conv2d],
                       weights_regularizer=regularizers.l2_regularizer(0.0001),
                       weights_initializer=initializers.variance_scaling_initializer(),
                       activation_fn=None,
                       biases_initializer=None):
            for i in range(order):
                out = 1.0
                for j in range(i+1):
                    out *= layers_lib.conv2d(inputs, inter_channels, 1, scope='hoa_conv_stage_1_order_%d_%d' % (i+1, j+1))
                out = tf.nn.relu(out)
                out = layers_lib.conv2d(out, in_channels, 1, scope='hoa_conv_stage_2_conv_order_%d' % (i+1))
                out = tf.nn.sigmoid(out)
                attention += out

        output = inputs * attention / order

    return output


def resnet50(image_input, is_training, embedding_dim, orders=3):
    out = resnet50_part1(image_input, is_training)

    final_outputs = 0

    flag = False
    for order in range(orders):
        order_output = high_order_module(out, order=order+1, scope='order_%d' % (order+1))
        order_output = resnet50_part2(order_output, is_training, embedding_dim, reuse=flag)
        final_outputs += order_output

        flag = True

    final_outputs /= orders

    return final_outputs


if __name__ == '__main__':
    with tf.Graph().as_default():
        image_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        is_training = tf.placeholder(dtype=tf.bool)
        output = resnet50(image_input, is_training, 128)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        sess.run(tf.global_variables_initializer())

        for var in tf.global_variables():
            print(var.name)

        restore_variables = [var for var in tf.trainable_variables() if 'resnet_v2_50' in var.name]
        saver = tf.train.Saver(var_list=restore_variables)
        saver.restore(sess=sess, save_path='../weights/pretrain/resnet50/resnet_v2_50.ckpt')

    print(output)