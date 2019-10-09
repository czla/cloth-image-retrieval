import tensorflow as tf


# easy positive loss, semi hard negative pair
def loss(inputs, labels):
    norm_inputs = tf.nn.l2_normalize(inputs, axis=1)

    dot_matrix = tf.expand_dims(norm_inputs, 1) * tf.expand_dims(norm_inputs, 0)
    sim_matrix = tf.reduce_sum(dot_matrix, axis=2)

    mask = tf.equal(tf.expand_dims(labels, axis=1), tf.expand_dims(labels, axis=0))
    mask = tf.cast(mask, dtype=tf.float32)
    pos_mask = mask - tf.matrix_diag(tf.ones_like(labels, dtype=tf.float32))
    neg_mask = 1.0 - mask

    easy_a_p = tf.reduce_max(sim_matrix * pos_mask - 1e10*(1.0-pos_mask), axis=1, keepdims=True)

    easy_a_n = tf.reduce_min(sim_matrix * neg_mask + 1e10*(1.0-neg_mask), axis=1, keepdims=True)
    easy_a_n_mask = tf.logical_and(tf.equal(sim_matrix, easy_a_n), tf.cast(neg_mask, dtype=tf.bool))
    sh_a_n_mask = tf.logical_and(tf.less(sim_matrix, easy_a_p), tf.cast(neg_mask, dtype=tf.bool))
    sh_a_n_mask = tf.logical_or(sh_a_n_mask, easy_a_n_mask)
    sh_a_n_mask = tf.cast(sh_a_n_mask, dtype=tf.float32)
    sh_a_n = tf.reduce_max(sim_matrix * sh_a_n_mask-1e10*(1-sh_a_n_mask), axis=1, keepdims=True)

    easy_a_p_exp = tf.exp(easy_a_p)
    sh_a_n_exp = tf.exp(sh_a_n)
    ep_loss = -tf.log(easy_a_p_exp / (easy_a_p_exp + sh_a_n_exp))

    ep_loss_mean = tf.reduce_mean(ep_loss)

    return ep_loss_mean


if __name__ == '__main__':
    import numpy as np

    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None,128))
        labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        loss = loss(inputs, labels)

        data = np.random.randn(6,128)
        l = np.array([0,0,0,1,1,1])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_output = sess.run(loss, feed_dict={inputs: data, labels: l})
        print(loss_output[0])
        print(loss_output[1])
        print(loss_output[2])
        print(loss_output[3])