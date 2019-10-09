import tensorflow as tf
import numbers
import math


def trihard_loss(distance_matrix, same_mask, margin):
    a_p_max = tf.reduce_max(distance_matrix*same_mask, axis=1)
    a_n_min = tf.reduce_min(distance_matrix+same_mask*1e10, axis=1)
    diff = a_p_max - a_n_min

    if isinstance(margin, numbers.Real):
        loss = tf.maximum(diff + margin, 0.0)
    elif margin == 'soft':
        loss = tf.nn.softplus(diff)
    else:
        raise NotImplementedError('The margin {} is not implemented in batch_hard'.format(margin))

    count = tf.count_nonzero(loss)
    count = tf.cast(count, tf.float32)

    loss_mean = tf.reduce_sum(loss) / count

    return loss_mean


def msml_loss(distance_matrix, same_mask, margin):
    a_p_max = tf.reduce_max(distance_matrix*same_mask)
    a_n_min = tf.reduce_min(distance_matrix+same_mask*1e10)
    diff = a_p_max - a_n_min

    if isinstance(margin, numbers.Real):
        loss = tf.maximum(diff + margin, 0.0)
    elif margin == 'soft':
        loss = tf.nn.softplus(diff)
    else:
        raise NotImplementedError('The margin {} is not implemented in batch_hard'.format(margin))

    return loss


def loss(inputs, labels, margin='soft', metric='euclidean'):
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

    trihard_loss_mean = trihard_loss(distance_matrix, same_mask, margin)
    msml_loss_mean = msml_loss(distance_matrix, same_mask, margin)

    if isinstance(margin, numbers.Real):
        loss = tf.where(trihard_loss_mean>(margin-0.05), trihard_loss_mean, msml_loss_mean)
    elif margin == 'soft':
        loss = tf.where(trihard_loss_mean>math.log(1+math.exp(-0.05)), trihard_loss_mean, msml_loss_mean)
    else:
        raise NotImplementedError('The margin {} is not implemented in batch_hard'.format(margin))

    return loss


if __name__ == '__main__':
    import numpy as np

    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None,2))
        labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        loss = loss(inputs, labels, margin=1.0, metric='euclidean')

        data = np.random.randn(6,2)
        l = np.array([0,0,1,1,2,2])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_output = sess.run(loss, feed_dict={inputs: data, labels: l})
        print(loss_output)
