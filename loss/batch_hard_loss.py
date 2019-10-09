import tensorflow as tf
import numbers


def loss(inputs, labels, margin='soft', metric='euclidean'):
    norm_inputs = tf.nn.l2_normalize(inputs, axis=1)
    sub_matrix = tf.expand_dims(norm_inputs, 1) - tf.expand_dims(norm_inputs, 0)
    if metric == 'sqeuclidean':
        distance_matrix = tf.reduce_sum(tf.square(sub_matrix), axis=-1) + 1e-12
    elif metric == 'euclidean':
        distance_matrix = tf.sqrt(tf.reduce_sum(tf.square(sub_matrix), axis=-1) + 1e-12)
    elif metric == 'cityblock':
        distance_matrix = tf.reduce_sum(tf.abs(sub_matrix), axis=-1)
    else:
        raise NotImplementedError('The following metric is not implemented by `cdist` yet: {}'.format(metric))

    mask = tf.equal(tf.expand_dims(labels, axis=1), tf.expand_dims(labels, axis=0))
    mask = tf.cast(mask, dtype=tf.float32)
    pos_mask = mask - tf.matrix_diag(tf.ones_like(labels, dtype=tf.float32))
    neg_mask = 1.0 - mask

    a_p_max = tf.reduce_max(distance_matrix*pos_mask, axis=1)
    a_n_min = tf.reduce_min(distance_matrix*neg_mask+1e10*(1.0-neg_mask), axis=1)
    diff = a_p_max - a_n_min

    if isinstance(margin, numbers.Real):
        loss = tf.maximum(diff + margin, 0.0)
    elif margin == 'soft':
        loss = tf.nn.softplus(diff)
    else:
        raise NotImplementedError('The margin {} is not implemented in batch_hard'.format(margin))

    count = tf.reduce_sum(tf.cast(tf.less(0., loss), tf.float32))

    loss_mean = tf.reduce_sum(loss) / (count + 1e-12)

    return loss_mean


if __name__ == '__main__':
    import numpy as np

    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None,2))
        labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        loss = loss(inputs, labels, margin=1.0)

        data = np.random.randn(6,2)
        l = np.array([0,0,1,1,2,2])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_output = sess.run(loss, feed_dict={inputs: data, labels: l})
        print(loss_output)