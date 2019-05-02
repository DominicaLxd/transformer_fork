

import tensorflow as tf
import numpy as np


def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
            # embedding第一行为全0，pad标志符   0 <pad>
    return embeddings


def positional_encoding(inputs, maxlen, masking=True, scope="positional_encoding"):
    """
    与embedding 相加

    :param inputs: [N, T, E] [128, ?, 512] [batch_size, len, d_model]
    :param maxlen: >= T
    :param masking:True, padding positions set to zeros
    :param scope:
    :return: [N, T, E]
    """
    N = tf.shape(inputs)[0]
    T = tf.shape(inputs)[1]
    E = inputs.get_shape().as_list()[-1]  # static
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # tf.tile 复制多次
        # tf.range(T) 0~T的序列 （T， ）
        # tf.expand_dims 扩展为rank1 （1， T）
        # tf.tile 0维度复制N次， 1维度不复制 （N, T）
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(maxlen)
        ])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def mask(inputs, queries=None, keys=None, type=None):

    """

    :param inputs:  [N , T_q, T_k]
    :param queries: [N , T_q, T_k]
    :param keys:    [N , T_q, T_k]
    :param type:
    :return:
    """
    # INT_MIN = -4294967296
    # 防止溢出？
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # keys 中为0。则为<pad>，pad为-MIN_VALUE, 做极小值，不对打分做影响
        # K(N * h, T_k, d_model/h)
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
        masks = tf.expand_dims(masks, 1) # [N, 1, T_k]
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)

        # 在encode中用

    elif type in ("q", "query", "queries"):
        # query 中为0。则为<pad>，pad为-MIN_VALUE, 做极小值，不对打分做影响
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        masks = tf.expand_dims(masks, -1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])

        # query
        outputs = inputs * masks

    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        # 三角阵
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)

    else:
        print("check correctly!")

    return outputs



def scaled_dot_product_attention(Q, K, V, causality=False, dropout_rate=0.3, training=True, scope="scaled_dot_product_attention"):
    """

    :param Q:   [N, T_q, d_k]
    :param K:   [N, T_k, d_k]
    :param V:   [N, T_k, d_v] d_v == d_k
    :param causality: 是否mask未来时间步
    :param dropout_rate: 0.3
    :param training:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # Q(N * h, T_q, d_model/h)
        # K(N * h, T_k, d_model/h) -> (N * h, d_model/h, T_k)
        # MatMul tf.matmul 只对最后两维做dot，第一维相同不管
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

        # [N * h, T_q, T_k]
        # scale
        outputs /= d_k ** 0.5

        # mask
        outputs = mask(outputs, Q, K, type="key")

        # mask future step
        if causality:
            outputs = mask(outputs, type='future')

        # softmax
        outputs = tf.nn.softmax(outputs)

        # [N * h, T_k, T_q]
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # mask Q
        outputs = mask(outputs, Q, K, type="query")

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        outputs = tf.matmul(outputs, V)

    return outputs

def multihead_attention(queries, keys, values, num_heads=8, dropout_rate=0.3, training=True, causality=False, scope='multihead_attention'):
    """
    multihead attention
    :param queries:  [N, T_q, d_model]
    :param keys:    [N, T_k, d_model]
    :param values:  [N, T_k, d_model] T_k == T_v v，k 相同
    :param num_heads:
    :param dropout_rate:
    :param training:
    :param causality:  是否mask当前时间步右侧
    :param scope:
    :return: （N, T_q, C）
    """

    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # tf.layers.dense 只改变最后一维
        Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

        # split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (N * h, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (N * h, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (N * h, T_k, d_model/h)

        # attention compate
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, d_model)

        # residual connect
        outputs += queries

        outputs = ln(outputs)

    return outputs

def ln(inputs, epsilon=1e-8, scope="ln"):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        # [-1:] 等价与 [-1]
        params_shape = inputs_shape[-1:]

        # 计算均值和方差，最后一维为模型dim
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())

        # variance 为方差，需要开方
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def feed_forward(inputs, num_units, scope="position_wise_feed_forward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])

        outputs += inputs
        outputs = ln(outputs)

    return outputs

def label_smoothing(inputs, epsilon=0.1):
    """

    :param inputs:
    :param epsilon:
    :return:
    """
    # 变得更soft
    V = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / V)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    #lrate = d−0.5 ·min(step_num−0.5, step_num·warmup_steps−1.5)
    # 论文里面的公式
    step = tf.cast(global_step+1, dtype=tf.float32)
    new_lr = init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
    return new_lr
