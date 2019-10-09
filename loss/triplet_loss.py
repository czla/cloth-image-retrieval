import tensorflow as tf


def triplet_loss(anchor, pos, neg, margin):
    a_p_distance = tf.reduce_sum(tf.square(anchor-pos), axis=1)
    a_n_distance = tf.reduce_sum(tf.square(anchor-neg), axis=1)

    loss = tf.maximum(margin+a_p_distance-a_n_distance, tf.zeros_like(a_p_distance))
    loss_mean = tf.reduce_mean(loss)

    return loss_mean
