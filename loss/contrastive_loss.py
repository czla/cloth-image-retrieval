import tensorflow as tf


def loss(inputs, labels, margin=1.0, metric='sqeuclidean'):
    norm_inputs = tf.nn.l2_normalize(inputs, axis=1)
    if metric == 'sqeuclidean':
        sub_matrix = tf.expand_dims(norm_inputs, 1) - tf.expand_dims(norm_inputs, 0)
        distance_matrix = tf.reduce_sum(tf.square(sub_matrix), axis=-1)
    elif metric == 'euclidean':
        sub_matrix = tf.expand_dims(norm_inputs, 1) - tf.expand_dims(norm_inputs, 0)
        distance_matrix = tf.sqrt(tf.reduce_sum(tf.square(sub_matrix), axis=-1))
    elif metric == 'cityblock':
        sub_matrix = tf.expand_dims(norm_inputs, 1) - tf.expand_dims(norm_inputs, 0)
        distance_matrix = tf.reduce_sum(tf.abs(sub_matrix), axis=-1)
    elif metric == 'cosine':
        sub_matrix = tf.expand_dims(norm_inputs, 1) * tf.expand_dims(norm_inputs, 0)
        distance_matrix = tf.reduce_sum(sub_matrix, axis=-1)
    else:
        raise NotImplementedError('The following metric is not implemented by `cdist` yet: {}'.format(metric))

    mask = tf.equal(tf.expand_dims(labels, axis=1), tf.expand_dims(labels, axis=0))
    mask = tf.cast(mask, dtype=tf.float32)
    pos_mask = mask - tf.matrix_diag(tf.ones_like(labels, dtype=tf.float32))
    neg_mask = 1.0 - mask

    if metric != 'cosine':
        pos_pair_loss = distance_matrix * pos_mask
        pos_pair_num = tf.cast(tf.count_nonzero(pos_mask), tf.float32)
        pos_pair_loss_mean = tf.reduce_sum(pos_pair_loss) / (2.0*pos_pair_num)

        neg_pair_loss = tf.nn.relu(margin-distance_matrix) * neg_mask
        neg_pair_num = tf.reduce_sum(tf.cast(tf.less(0., neg_pair_loss), tf.float32))
        neg_pair_loss_mean = tf.reduce_sum(neg_pair_loss) / (2.0 * neg_pair_num + 1e-12)
    else:
        pos_pair_loss = tf.square(distance_matrix - 1.0) * pos_mask
        pos_pair_num = tf.cast(tf.count_nonzero(pos_mask), tf.float32)
        pos_pair_loss_mean = tf.reduce_sum(pos_pair_loss) / (2.0*pos_pair_num)

        neg_pair_loss = tf.nn.relu(distance_matrix - margin) * neg_mask
        neg_pair_num = tf.reduce_sum(tf.cast(tf.less(0., neg_pair_loss), tf.float32))
        neg_pair_loss_mean = tf.reduce_sum(neg_pair_loss) / (2. * neg_pair_num + 1e-12)

    loss_mean = pos_pair_loss_mean + neg_pair_loss_mean

    return loss_mean


if __name__ == '__main__':
    import numpy as np

    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None,512))
        labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        loss = loss(inputs, labels, margin=0.5, metric='cosine')

        data = np.random.randn(6,512)
        l = np.array([0,0,1,1,2,2])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_output = sess.run(loss, feed_dict={inputs: data, labels: l})
        print(loss_output)