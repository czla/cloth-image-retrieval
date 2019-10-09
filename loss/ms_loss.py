import tensorflow as tf


def loss(embeddings, labels, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, ms_mining=False):
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    labels = tf.reshape(labels, [-1, 1])

    batch_size = embeddings.get_shape().as_list()[0]

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat-(1-mask_neg)*1e10, axis=1, keepdims=True)
        min_val = tf.reduce_min(pos_mat+(1-mask_pos)*1e10, axis=1, keepdims=True)

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb)) * mask_pos
    neg_exp = tf.exp(beta * (neg_mat - lamb)) * mask_neg

    pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term)

    return loss
