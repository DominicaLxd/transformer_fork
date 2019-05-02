import tensorflow as tf
from hparams import hparams

def load_data(fpath1, fpath2, maxlen1, maxlen2):

    sents1, sents2 = [], []

    with open(fpath1, 'r') as f1, open(fpath2, 'r') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1:
                continue
            if len(sent2.split()) + 1 > maxlen2:
                continue
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())

    return sents1, sents1

def load_vocab(vocab_fpath):
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token

def encode(input, type, dict):
    input_str = input.decode("utf-8")
    if type == "x":
        tokens = input_str.split() + ["</s>"]
    else:
        tokens = ["<s>"] + input_str.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x


def generator_fn(sents1, sents2, vocab_fpath):
    token2idx, idx2token = load_vocab(vocab_fpath)

    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        # 不要最后一位和不要第一位， 第一位是开始符号
        decoder_input, y = y[:-1], y[1:]
        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)

def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):
    """
    :param sents1: list of source
    :param sents2: list of target
    :param vocab_fpath: string.
    :param batch_size:
    :param shuffle:
    :return:xs tuple of
                x: int32 tensor (N, T1)
                x_seqlens: int32 tensor(N, )
                sents1: str tensor (N, )
            ys tuple of
                decoder_input: int32 tensor (N, T2)
                y: int32 tensor (N, T2)
                y_seqlen: int32 tensor (N, )
                sents2: str tensor(N, )
    """
    shapes = (
        ([None], (), ()),
        ([None], [None], (), ())
    )
    types = (
        (tf.int32, tf.int32, tf.string),
        (tf.int32, tf.int32, tf.int32, tf.string)
    )
    paddings = (
        (0, 0, ''),
        (0, 0, 0, '')
    )
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath)  # generator_fn 参数
    )

    if shuffle:
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset



def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):

    # 读取metadata
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    #num_batches = calc_num_batches(len(sents1), batch_size)
    num_batches = (len(sents1) // batch_size + int(len(sents1) % batch_size != 0))
    return batches, num_batches, len(sents1)

# test
"""
if __name__ == '__main__':
    hp = hparams
    
     # 0 <pad>
    # 1 <unk>
    # 2 <s>
    # 3 </s>
    train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2, hp.maxlen_source, hp.maxlen_target,
                                                                    hp.vocab, hp.batch_size, shuffle=True)

    eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2, 100000, 100000, hp.vocab,
                                                                 hp.batch_size, shuffle=False)
    
    #get_batch(hp.train1, hp.train2, hp.maxlen_source, hp.maxlen_target, hp.vocab, hp.batch_size, shuffle=True)
    #get_batch(hp.eval1, hp.eval2, 100000, 100000, hp.vocab, hp.batch_size, shuffle=False)
    sents1, sents2 = load_data(hp.train1, hp.train2, hp.maxlen_source, hp.maxlen_target)
    token2idx, idx2token = load_vocab(hp.vocab)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        print(x)
        print()
        print(y)
        exit(0)
    """
