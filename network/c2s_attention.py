import tensorflow as tf
import numpy as np


def residual_module(x, filters, strides, is_training, name, shortcut=True):
    expand_dim = 4

    out = tf.layers.conv2d(inputs=x, filters=filters//expand_dim, kernel_size=1, strides=strides, padding='same', use_bias=False, name='res%s_branch2a' % name)
    out = tf.layers.batch_normalization(inputs=out, training=is_training, name='bn%s_branch2a' % name)
    out = tf.nn.relu(out)
    out = tf.layers.conv2d(inputs=out, filters=filters//expand_dim, kernel_size=3, strides=1, padding='same', use_bias=False, name='res%s_branch2b' % name)
    out = tf.layers.batch_normalization(inputs=out, training=is_training, name='bn%s_branch2b' % name)
    out = tf.nn.relu(out)
    out = tf.layers.conv2d(inputs=out, filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False, name='res%s_branch2c' % name)
    out = tf.layers.batch_normalization(inputs=out, training=is_training, name='bn%s_branch2c' % name)

    if shortcut:
        branch = x
    else:
        branch = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=1, strides=strides, padding='same', use_bias=False, name='res%s_branch1' % name)
        branch = tf.layers.batch_normalization(inputs=branch, training=is_training, name='bn%s_branch1' % name)

    out = tf.add(out, branch)
    out = tf.nn.relu(out)

    return out


def residule_block(x, filters, strides, is_training, names):
    out = residual_module(x, filters, strides, is_training, names[0], shortcut=False)

    for i in range(1,len(names)):
        out = residual_module(out, filters, 1, is_training, names[i])

    return out


def resnet50(image_input, is_training):
    stage1 = tf.layers.conv2d(inputs=image_input, filters=64, kernel_size=7, strides=2, padding='same', name='conv1')
    stage1 = tf.layers.batch_normalization(inputs=stage1, training=is_training, name='bn_conv1')
    stage1 = tf.nn.relu(stage1)

    stage2 = tf.layers.max_pooling2d(inputs=stage1, pool_size=3, strides=2, padding='same', name='pool1')
    stage2 = residule_block(stage2, 256, 1, is_training, ['2a','2b','2c'])

    stage3 = residule_block(stage2, 512, 2, is_training, ['3a', '3b', '3c', '3d'])

    out = tf.layers.conv2d(inputs=stage3, filters=1, kernel_size=3, strides=1, padding='same', use_bias=False, name='get_depth')
    out = tf.sigmoid(out)

    return out


def inception_module(x, filters_list, name, suffix=''):
    inception_1x1 = tf.layers.conv2d(inputs=x, filters=filters_list[0][0], kernel_size=1, strides=1, padding='same', name='inception_%s_1x1%s' % (name,suffix))
    inception_1x1 = tf.nn.relu(inception_1x1)

    inception_3x3 = tf.layers.conv2d(inputs=x, filters=filters_list[1][0], kernel_size=1, strides=1, padding='same', name='inception_%s_3x3_reduce%s' % (name,suffix))
    inception_3x3 = tf.nn.relu(inception_3x3)
    inception_3x3 = tf.layers.conv2d(inputs=inception_3x3, filters=filters_list[1][1], kernel_size=3, strides=1, padding='same', name='inception_%s_3x3%s' % (name,suffix))
    inception_3x3 = tf.nn.relu(inception_3x3)

    inception_5x5 = tf.layers.conv2d(inputs=x, filters=filters_list[2][0], kernel_size=1, strides=1, padding='same', name='inception_%s_5x5_reduce%s' % (name,suffix))
    inception_5x5 = tf.nn.relu(inception_5x5)
    inception_5x5 = tf.layers.conv2d(inputs=inception_5x5, filters=filters_list[2][1], kernel_size=5, strides=1, padding='same', name='inception_%s_5x5%s' % (name,suffix))
    inception_5x5 = tf.nn.relu(inception_5x5)

    inception_pool = tf.layers.max_pooling2d(inputs=x, pool_size=3, strides=1, padding='same', name='inception_%s_pool%s' % (name,suffix))
    inception_pool = tf.layers.conv2d(inputs=inception_pool, filters=filters_list[3][0], kernel_size=1, strides=1, padding='same', name='inception_%s_pool_proj%s' % (name,suffix))
    inception_pool = tf.nn.relu(inception_pool)

    inception_out = tf.concat([inception_1x1,inception_3x3,inception_5x5,inception_pool], axis=3)

    return inception_out


def inception_layers(x, is_training, suffix=''):
    out = inception_module(x, [[64], [96, 128], [16, 32], [32]], '3a', suffix)
    out = inception_module(out, [[128], [128, 192], [32, 96], [64]], '3b', suffix)
    out = tf.layers.max_pooling2d(inputs=out, pool_size=3, strides=2, padding='same', name='pool3_3x3_s2')

    out = inception_module(out, [[192], [96, 208], [16, 48], [64]], '4a', suffix)
    out = inception_module(out, [[160], [112, 224], [24, 64], [64]], '4b', suffix)
    out = inception_module(out, [[128], [128, 256], [24, 64], [64]], '4c', suffix)
    out = inception_module(out, [[112], [144, 288], [32, 64], [64]], '4d', suffix)
    out = inception_module(out, [[256], [160, 320], [32, 128], [128]], '4e', suffix)
    out = tf.layers.max_pooling2d(inputs=out, pool_size=3, strides=2, padding='same', name='pool4_3x3_s2')

    out = inception_module(out, [[256], [160, 320], [32, 128], [128]], '5a', suffix)
    out = inception_module(out, [[384], [192, 384], [48, 128], [128]], '5b', suffix)

    out = tf.reduce_mean(input_tensor=out, axis=[1,2], name='avg_%s' % suffix)

    out = tf.layers.dropout(inputs=out, rate=0.5, training=is_training, name='loss_drop%s' % suffix)
    out = tf.layers.dense(inputs=out, units=128, name='loss3_fc%s' % suffix)
    out = tf.nn.l2_normalize(out, axis=1, name='norm%s' % suffix)

    return out


def googlenet(image_input, is_training):
    conv_1 = tf.layers.conv2d(inputs=image_input, filters=64, kernel_size=7, strides=2, padding='same', name='conv1_7x7_s2')
    conv_1 = tf.nn.relu(conv_1)
    conv_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=3, strides=2, padding='same', name='pool1_3x3_s2')
    conv_1 = tf.nn.local_response_normalization(input=conv_1, depth_radius=2, alpha=1.99999994948e-05, beta=0.75, name='pool1_norm1')

    conv_2 = tf.layers.conv2d(inputs=conv_1, filters=64, kernel_size=1, strides=1, padding='same', name='conv2_3x3_reduce')
    conv_2 = tf.nn.relu(conv_2)

    conv_3 = tf.layers.conv2d(inputs=conv_2, filters=192, kernel_size=3, strides=1, padding='same', name='conv2_3x3')
    conv_3 = tf.nn.relu(conv_3)
    conv_3 = tf.nn.local_response_normalization(input=conv_3, depth_radius=2, alpha=1.99999994948e-05, beta=0.75, name='conv2_norm2')
    conv_3 = tf.layers.max_pooling2d(inputs=conv_3, pool_size=3, strides=2, padding='same', name='pool2_3x3_s2')

    out = inception_layers(conv_3, is_training)

    attention_map = resnet50(image_input, is_training)
    conv_3_attention = tf.multiply(conv_3, attention_map)
    att_out = inception_layers(conv_3_attention, is_training, suffix='_p')

    match_output = tf.concat([out, att_out], axis=-1, name='sum_main_branch')

    return match_output


def load_weights(sess, model_path):
    data_dict = np.load(model_path, encoding='bytes').item()

    print('load weights')
    for op_name in data_dict:
        with tf.variable_scope(op_name, reuse=True):
            for name, data in data_dict[op_name].items():
                param_name = name.decode()
                if param_name == 'weights':
                    param_name = 'kernel'
                elif param_name == 'biases':
                    param_name = 'bias'
                elif param_name == 'variance':
                    param_name = 'moving_variance'
                elif param_name == 'mean':
                    param_name = 'moving_mean'
                elif param_name == 'scale':
                    param_name = 'gamma'
                elif param_name == 'offset':
                    param_name = 'beta'
                print('variable %s/%s' % (op_name, param_name))
                var = tf.get_variable(param_name)
                sess.run(var.assign(data))


if __name__ == '__main__':
    import cv2
    from tqdm import tqdm
    from tensorflow.python.framework import graph_util

    def image_preprocess(image, bgr=True):
        image = image.astype(np.float32)
        mean = [119.6200, 120.8229, 104.2657]  # rgb
        if bgr:
            mean = mean[::-1]
        image_mean = np.asarray(mean, dtype=np.float32)
        image = image - image_mean

        return image

    def load_image(im_path):
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (192, 192))
        im = image_preprocess(im)

        return im

    with tf.Graph().as_default():
        image_input = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        outputs = googlenet(image_input, False)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())

        load_weights(sess, '../weights/pretrain/c2s_attention/pretrain_c2s_attention.npy')

        data = np.random.randn(2,224,224,3)
        out = sess.run(outputs, feed_dict={image_input: data})
        print(out.shape)
        print(np.linalg.norm(out, axis=1))

        # consumer_data = open('data/DeepFashion/testConsumer.txt').readlines()
        # consumer_data = list(map(lambda line: line.strip().split('\t'), consumer_data))
        # consumer_images = list(map(lambda line: line[0], consumer_data))
        # consumer_shop_indexes = list(map(lambda line: int(line[1]), consumer_data))
        #
        # consumer_features = []
        # for consumer_image in tqdm(consumer_images):
        #     image = load_image(consumer_image)
        #     image = np.expand_dims(image, axis=0)
        #
        #     feature = sess.run(match_output, feed_dict={image_input: image, is_training: False})[0]
        #     consumer_features.append(feature)
        # consumer_features = np.array(consumer_features)
        # consumer_shop_indexes = np.array(consumer_shop_indexes)
        #
        # shop_images = open('data/DeepFashion/testShop.txt').readlines()
        # shop_images = list(map(lambda line: line.strip(), shop_images))
        # shop_features = []
        # for shop_image in tqdm(shop_images):
        #     image = load_image(shop_image)
        #     image = np.expand_dims(image, axis=0)
        #
        #     feature = sess.run(match_output, feed_dict={image_input: image, is_training: False})[0]
        #     shop_features.append(feature)
        # shop_features = np.array(shop_features)
        #
        # np.save('results/c2s_attention_old/consumer_features.npy', consumer_features)
        # np.save('results/c2s_attention_old/consumer_shop_indexes.npy', consumer_shop_indexes)
        # np.save('results/c2s_attention_old/shop_features.npy', shop_features)