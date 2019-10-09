import tensorflow as tf


def loss(inputs, labels, margin, beta, metric='euclidean'):
    sub_matrix = (tf.expand_dims(inputs, 1) - tf.expand_dims(inputs, 0))
    if metric == 'sqeuclidean':
        distance_matrix = tf.reduce_sum(tf.square(sub_matrix), axis=-1) + 1e-12
    elif metric == 'euclidean':
        distance_matrix = tf.sqrt(tf.reduce_sum(tf.square(sub_matrix), axis=-1) + 1e-12)
    elif metric == 'cityblock':
        distance_matrix = tf.reduce_sum(tf.abs(sub_matrix), axis=-1)
    else:
        raise NotImplementedError('The following metric is not implemented by `cdist` yet: {}'.format(metric))

    same_mask = tf.equal(tf.expand_dims(labels, axis=1), tf.expand_dims(labels, axis=0))
    same_mask = tf.cast(same_mask, tf.float32)

    a_p = tf.reduce_max(distance_matrix*same_mask, axis=1)
    a_n = tf.reduce_min(distance_matrix+same_mask*1e10, axis=1)

    a_p_loss = tf.maximum(margin+a_p-beta, 0.0)
    a_n_loss = tf.maximum(margin-a_n+beta, 0.0)

    count = tf.count_nonzero(a_p_loss) + tf.count_nonzero(a_n_loss)
    count = tf.cast(count, tf.float32)

    loss_sum = tf.reduce_sum(a_p_loss) + tf.reduce_sum(a_n_loss)
    loss_mean = loss_sum / count

    return loss_mean


if __name__ == '__main__':
    import numpy as np

    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None,2))
        labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        loss = loss(inputs, labels, margin=0.2, beta=1.2)

        data = np.random.randn(6,2)
        l = np.array([0,0,1,1,2,2])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_output = sess.run(loss, feed_dict={inputs: data, labels: l})
        print(loss_output)
