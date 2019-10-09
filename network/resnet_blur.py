import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops, nn_ops


import numpy as np


def blur_pool(x, filt_size=7, stride=2, pad_off=0):
    channels = utils.last_dimension(x.get_shape(), min_rank=4)

    if (filt_size == 1):
        a = np.array([1., ])
    elif (filt_size == 2):
        a = np.array([1., 1.])
    elif (filt_size == 3):
        a = np.array([1., 2., 1.])
    elif (filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif (filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif (filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif (filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    kernel_1 = a[:,None]*a[None,:]
    kernel_1 = np.expand_dims(kernel_1/np.sum(np.sum(kernel_1)), -1)

    kernel_1 = np.expand_dims(kernel_1.repeat(axis=-1, repeats=channels), axis=-1).repeat(axis=-1, repeats=1).astype(np.float32)
    kernel = tf.Variable(kernel_1, trainable=False)


    pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                      int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
    pad_sizes = [pad_size + pad_off for pad_size in pad_sizes]
    l1 = list(pad_sizes[:2])
    l2 = list(pad_sizes[2:])

    if filt_size == 1:
        if pad_off==0:
            return x[:,:,::stride,::stride]
        else:
            return tf.pad(x, [[0,0],l1,l2,[0,0]], 'REFLECT')[:,:,::stride,::stride]
    else:

        x = tf.pad(x, [[0,0],l1,l2,[0,0]], 'REFLECT')
        x = tf.nn.depthwise_conv2d(x, kernel, strides=[1,stride,stride,1],padding='VALID')
        return x


def max_pool(inputs, kernel_size, stride=1, scope=None):
    if kernel_size == 1 and stride == 1:
        outputs = inputs
    else:
        outputs = layers.max_pool2d(inputs=inputs, kernel_size=kernel_size, stride=1, scope=scope)
        if stride > 1:
            outputs = blur_pool(outputs, stride=stride)

    return outputs


def conv2d(inputs, num_outputs, kernel_size, stride=1, rate=1, scope=None):
    # if stride == 1 or kernel_size == 1:
    #     return layers_lib.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    # else:
    #     kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    #     pad_total = kernel_size_effective - 1
    #     pad_beg = pad_total // 2
    #     pad_end = pad_total - pad_beg
    #     inputs = array_ops.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    #     return layers_lib.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)
    outputs = layers_lib.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    if stride > 1:
        outputs = blur_pool(outputs, stride=stride)

    return outputs


def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1):
    with tf.variable_scope('bottleneck_v2'):
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = layers.batch_norm(inputs, activation_fn=nn_ops.relu, scope='preact')
        if depth == depth_in:
            shortcut = max_pool(inputs=inputs, kernel_size=1, stride=stride, scope='shortcut')
        else:
            with arg_scope([layers_lib.conv2d], normalizer_fn=None, activation_fn=None):
                shortcut = conv2d(preact, depth, 1, stride=stride, scope='shortcut')

        residual = layers_lib.conv2d(preact, depth_bottleneck, 1, stride=1, scope='conv1')
        residual = conv2d(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = layers_lib.conv2d(residual, depth, 1, stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

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


def resnet50(image_input, is_training, scope='resnet_v2_50'):
    with tf.variable_scope(scope):
        with arg_scope([layers.batch_norm], is_training=is_training, scale=True):
            with arg_scope([layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
                out = conv2d(inputs=image_input, num_outputs=64, kernel_size=7, stride=2, scope='conv1')
            out = max_pool(inputs=out, kernel_size=3, stride=2, scope=scope)
            with arg_scope([layers_lib.conv2d], activation_fn=nn_ops.relu, normalizer_fn=layers.batch_norm):
                out = resnet_v2_block(inputs=out, base_depth=64, num_units=3, stride=2, scope='block1')
                out = resnet_v2_block(inputs=out, base_depth=128, num_units=4, stride=2, scope='block2')
                out = resnet_v2_block(inputs=out, base_depth=256, num_units=6, stride=2, scope='block3')
                out = resnet_v2_block(inputs=out, base_depth=512, num_units=3, stride=1, scope='block4')
            out = layers.batch_norm(out, activation_fn=nn_ops.relu, scope='postnorm')
            out = tf.reduce_mean(out, axis=[1,2], name='pool5')

    out = tf.layers.dense(inputs=out, units=128)
    out = tf.nn.l2_normalize(out, axis=1)

    return out


if __name__ == '__main__':
    with tf.Graph().as_default():
        image_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        is_training = tf.placeholder(dtype=tf.bool)
        output = resnet50(image_input, is_training)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        sess.run(tf.global_variables_initializer())

        for var in tf.all_variables():
            print(var.name)

        restore_variables = [var for var in tf.trainable_variables() if 'dense' not in var.name]
        saver = tf.train.Saver(var_list=restore_variables)
        saver.restore(sess=sess, save_path='../weights/pretrain/resnet50/resnet_v2_50.ckpt')

    print(output)
