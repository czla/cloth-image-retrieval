import tensorflow as tf


def loss(inputs, labels, embedding_dim, K, loss_func, margin='soft', metric='euclidean', hidden_dim=8192):
    with tf.variable_scope('horde_loss'):
        total_loss = []
        out = 1.0
        for order in range(K):
            with tf.variable_scope('order_%d' % (order+1)):
                out *= tf.layers.conv2d(inputs, hidden_dim, 1, use_bias=False, kernel_initializer=tf.glorot_uniform_initializer())
                if order > 0:
                    out_avg = tf.reduce_mean(out, axis=[1,2])
                    fc_out = tf.layers.dense(inputs=out_avg, units=embedding_dim, use_bias=False)
                    order_loss = loss_func.loss(fc_out, labels, margin=margin, metric=metric)
                    total_loss.append(order_loss)

    return total_loss


if __name__ == '__main__':
    import numpy as np
    import contrastive_loss

    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None,7,7,2048))
        labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        loss = loss(inputs, labels, 512, 4, contrastive_loss, 0.5, 'cosine')

        data = np.random.randn(6,7,7,2048)
        l = np.array([0,0,1,1,2,2])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for var in tf.global_variables():
            print(var.name)

        loss_output = sess.run(loss, feed_dict={inputs: data, labels: l})
        print(loss_output)