import argparse


class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    # # parser.add_argument('--vocab_size', default=32000, type=int)
    # parser.add_argument('--vocab_size', default=2048, type=int)
    parser.add_argument('--embed_size', default=52, type=int)

    # Preprocessed files
    default_dir = 'dataset/Flickr8k/prepro'
    parser.add_argument('--train', default='%s/train_set.pkl' % default_dir)
    parser.add_argument('--dev', default='%s/dev_set.pkl' % default_dir)
    parser.add_argument('--test', default='%s/train_set.pkl' % default_dir)
    parser.add_argument('--vec2word', default='%s/vec2word_model.npy' % default_dir)
    parser.add_argument('--split_size', default=1.0, type=float,
                        help="percentage [0,1] of the dataset that is randomly picked and used")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # Model
    # NOTE Kasper
    # feedforwards: these parameters are defined separately but d_model must be the same for both I think(??)

    # Encoder feedforward
    parser.add_argument('--d_model_enc', default=52, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff_enc', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    # Transformer feedforward
    parser.add_argument('--d_model_trans', default=52, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff_trans', default=2048, type=int,
                        help="hidden dimension of feedforward layer")

    # Transformer model parameters
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    # Num heads must cleanly divided --embed_size ( embed_size % num_heads == 0 )
    parser.add_argument('--num_heads', default=4, type=int,
                        help="number of attention heads")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    parser.add_argument('--maxlen1', default=34, type=int, help="maximum length of a input sequences")
    parser.add_argument('--maxlen2', default=50, type=int, help="maximum length of a output sequence")

    # test
    # parser.add_argument('--test1', default='iwslt2016/segmented/test.de.bpe',
    #                     help="german test segmented data")
    # parser.add_argument('--test2', default='iwslt2016/prepro/test.en',
    #                     help="english test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")
