import tensorflow as tf
from tensorflow.contrib import layers as layers_lib


def loss(embeddings, labels, num_classes):
    with tf.variable_scope('softmax_loss'):
        logits = layers_lib.fully_connected(embeddings, num_classes, activation_fn=None, biases_initializer=None)

        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        cls_loss_mean = tf.reduce_mean(cls_loss)

    return cls_loss_mean


if __name__ == '__main__':
    import numpy as np

    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None,2048))
        labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        loss = loss(inputs, labels, 10)

        data = np.random.randn(6,2048)
        l = np.array([0,0,1,1,2,2])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_output = sess.run(loss, feed_dict={inputs: data, labels: l})
        print(loss_output)