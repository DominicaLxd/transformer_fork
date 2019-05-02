import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # data files
    parser.add_argument('--train1', default='iwslt2016/segmented/train.de.bpe',
                        help="german training segmented data")
    parser.add_argument('--train2', default='iwslt2016/segmented/train.en.bpe',
                        help="english training segmented data")
    parser.add_argument('--eval1', default='iwslt2016/segmented/eval.de.bpe',
                        help="german evaluation segmented data")
    parser.add_argument('--eval2', default='iwslt2016/segmented/eval.en.bpe',
                        help="english evaluation segmented data")
    parser.add_argument('--eval3', default='iwslt2016/prepro/eval.en',
                        help="english evaluation unsegmented data")
    parser.add_argument('--vocab', default='iwslt2016/segmented/bpe.vocab',
                        help="vocabulary file path")

    # training scheme
    parser.add_argument('--vocab_size', default=32000, type=int)
    parser.add_argument('--lr', default=0.0003, type=float, help="default learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--logdir', default="log/1")

    # model
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--maxlen_source', default=100, type=int, help='max length of source')
    parser.add_argument('--maxlen_target', default=100, type=int, help='max length of target')
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--encode_num_blocks', default=6, type=int)
    parser.add_argument('--decode_num_blocks', default=6, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--forward_hidden', default=2048, type=int)



hparams = Hparams()
parser = hparams.parser
hparams = parser.parse_args()

