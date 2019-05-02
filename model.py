
import tensorflow as tf
import logging
from tqdm import tqdm
from data_load import load_vocab
from modules import *
from hparams import hparams
from utils import *

logging.basicConfig(level=logging.INFO)

class Transformer:
    """
    xs: tuple
        x: (N, T1) int32 embedding
        x_seqlens: (N, ) int32
        sents1: (N, ) str

    ys: tuple
        decoder_input: (N, T2) int32 self-attention input
        y: (N, T2) int32
        y_seqlen: (N, ) int32
        sents2: (N, ) str
    """
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)
        # (vocab_size, embedding_dim) = (32000, 512)

    def encode(self, xs, training=True):
        """
        memory: encoder outputs (N, T1, d_model)
        :param xs: tuple(x, seqlens, sents1)
        :param training:
        :return:

        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs
            # x shape = [N, T_x]
            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x)
            # [128, ?, 512] [batch_size, T_x, d_model]

            # multiply those weights by sqrt(d_model) 512
            enc *= self.hp.d_model ** 0.5

            # 位置编码 与enc直接相加
            enc += positional_encoding(enc, self.hp.maxlen_source)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            # enc_shape = [128, ?, 512] [batch_size, len, d_model] [N, T, E]
            # encode blocks
            for i in range(self.hp.encode_num_blocks):
                # 6层
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention， k=v=q
                    enc = multihead_attention(
                        queries=enc,
                        keys=enc,
                        values=enc,
                        num_heads=self.hp.num_heads,
                        dropout_rate=self.hp.dropout_rate,
                        training=training,
                        causality=False
                    )
                    enc = feed_forward(enc, num_units=[self.hp.forward_hidden, self.hp.d_model])
        memory = enc
        return memory, sents1

    def decode(self, ys, memory, training=True):
        """
        memory: (N, T1, d_model)
        :param ys:
        :param memory:
        :param training:
        :return:
        logits: (N, T2, V) float32 prob
        y_hat: (N, T2) int32
        y: (N, T2) int32
        sents2(N, ) string
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)
            dec *= self.hp.d_model ** 0.5

            dec += positional_encoding(dec, self.hp.maxlen_target)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            for i in range(self.hp.decode_num_blocks):
                with tf.variable_scope("num_block_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self_attention, mask 未来时间步
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # 正常attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")

                    dec = feed_forward(dec, num_units=[self.hp.forward_hidden, self.hp.d_model])
                    # [N, T2, d_model]

        # embedd 解码过程
        # [vocab_size, d_model] -> [d_model, vocab_size]
        weights = tf.transpose(self.embeddings)
        # [N, T2, vocab_size]
        logits = tf.einsum('ntd,dk->ntk', dec, weights)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        # y_hat为onehot
        return logits, y_hat, y, sents2

    def train(self, xs, ys):

        # forward
        memory, sents1 = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory)

        # train scheme
        # 对y做平滑操作
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        # 交叉熵损失
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        # 去除y的pad
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))
        # 求损失
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        # 学习率变化
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('global_step', global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        """
        自回归合成
        xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
        ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
        :return:
        """
        decoder_inputs, y, y_seqlen, sents2 = ys
        # [batch_size, 1] 1位为<s> 开始符 构造一个空白的输入，batch_size 和xs相同
        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)

        logging.info("Inference graph is being built, Please be patient.")
        #  真实场景中不知道maxlen_target 如何来，根据数据集的maxlen切分给出的
        for _ in tqdm(range(self.hp.maxlen_target)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, False)
            # 当预测y_hat为pad的值，为0 的时候，结束
            if tf.reduce_sum(y_hat, 1) == self.token2idx["pad"]:
                break

            # [N, T]
            # 构造新的input
            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        # 监视随机样本
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

"""
if __name__ == '__main__':
    xs = (tf.placeholder(dtype=tf.int32, shape=(hparams.batch_size, None), name='x'),
          tf.placeholder(dtype=tf.int32, shape=(hparams.batch_size,), name='x_seqlens'),
          tf.placeholder(dtype=tf.string, shape=(hparams.batch_size,), name='sents1'),
          )

    ys = (tf.placeholder(dtype=tf.int32, shape=(hparams.batch_size, None), name='decoder_input'),
          tf.placeholder(dtype=tf.int32, shape=(hparams.batch_size, None), name='y'),
          tf.placeholder(dtype=tf.int32, shape=(hparams.batch_size,), name='y_seqlens'),
          tf.placeholder(dtype=tf.string, shape=(hparams.batch_size,), name='sents2'),
          )
    model = Transformer(hparams)
    loss, train_op, global_step, summaries = model.train(xs, ys)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([loss, train_op, global_step, summaries])
"""
