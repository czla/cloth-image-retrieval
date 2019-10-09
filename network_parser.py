import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers

from network import resnet, resnet_attention


backbone_dict = {
    'resnet50': resnet.resnet50,
    'resnet50_attention': resnet_attention.resnet50,
}

backbone_scope = {
    'resnet50': 'resnet_v2_50',
    'resnet50_attention': 'resnet_v2_50',
}


def create_network(image_inputs, is_training, backbone, embedding_dim):
    out = backbone_dict[backbone](image_inputs, is_training)

    avg_out = tf.reduce_mean(out, axis=[1, 2])
    emb_out = layers_lib.fully_connected(avg_out, embedding_dim, activation_fn=None, weights_regularizer=regularizers.l2_regularizer(0.0001), biases_initializer=None)

    return out, emb_out


def load_backbone(sess, backbone_config):
    print('load backbone weights')

    scope = backbone_scope[backbone_config['name']]

    restore_variables = [var for var in tf.trainable_variables() if scope in var.name]
    restore_saver = tf.train.Saver(var_list=restore_variables)
    restore_saver.restore(sess=sess, save_path=backbone_config['pretrain'])