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


def resnet50(image_input, is_training, scope='resnet_v2_50'):
    with tf.variable_scope(scope):
        with arg_scope(resnet_arg_scope(is_training=is_training)):
            with arg_scope([layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
                stage1 = conv2d_same(inputs=image_input, num_outputs=64, kernel_size=7, stride=2, scope='conv1')
            stage2 = layers.max_pool2d(stage1, [3, 3], stride=2, scope='pool1')
            stage3 = resnet_v2_block(inputs=stage2, base_depth=64, num_units=3, stride=2, scope='block1')

    with tf.variable_scope('attention'):
        with arg_scope([layers_lib.conv2d], activation_fn=None, normalizer_fn=None, biases_initializer=None):
            c = stage3.shape[3]
            attention = layers_lib.conv2d(stage3, c//4, 1, activation_fn=nn_ops.relu)
            attention = layers_lib.conv2d(attention, c, 1)
            attention = tf.sigmoid(attention)
    stage3 *= attention

    with tf.variable_scope(scope):
        with arg_scope(resnet_arg_scope(is_training=is_training)):
            stage4 = resnet_v2_block(inputs=stage3, base_depth=128, num_units=4, stride=2, scope='block2')
            stage5 = resnet_v2_block(inputs=stage4, base_depth=256, num_units=6, stride=2, scope='block3')
            stage5 = resnet_v2_block(inputs=stage5, base_depth=512, num_units=3, stride=1, scope='block4')
            stage5 = layers.batch_norm(stage5, activation_fn=nn_ops.relu, scope='postnorm')

    return stage5


if __name__ == '__main__':
    with tf.Graph().as_default():
        image_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        is_training = tf.placeholder(dtype=tf.bool)
        output = resnet50(image_input, is_training)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        sess.run(tf.global_variables_initializer())

        for var in tf.global_variables():
            print(var.name)

        restore_variables = [var for var in tf.trainable_variables() if 'resnet_v2_50' in var.name]
        saver = tf.train.Saver(var_list=restore_variables)
        saver.restore(sess=sess, save_path='../pretrain/resnet50/resnet_v2_50.ckpt')

    print(output)