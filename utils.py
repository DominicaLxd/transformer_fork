import tensorflow as tf
import logging
import os
import re

logging.basicConfig(level=logging.INFO)

def save_variable_specs(fpath):
    def _get_size(shape):
        size = 1
        for d in range(len(shape)):
            size *= shape[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}=={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params:", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))

    logging.info("Variables info has been saved.")


def convert_idx_to_token_tensor(inputs, idx2token):
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    # hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)
    """

    :param num_batches:
    :param num_samples:
    :param sess:
    :param tensor:
    :param dict:
    :return:
    """
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples]


def postprocess(hypotheses, idx2token):
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ") # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses

def calc_bleu(ref, translation):
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)

    except: pass
    os.remove("temp")