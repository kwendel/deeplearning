import os
import pickle
import random

import tensorflow as tf

from models.encoderdecoder import EncoderDecoder
from utils.hparams import Hparams
from utils.utils import calc_num_batches


# def load_vocab(vocab_fpath):
#     '''Loads vocabulary file and returns idx<->token maps
#     vocab_fpath: string. vocabulary file path.
#     Note that these are reserved
#     0: <s>, 1: <unk>, 2: <pad>, 3:</s>
#
#     Returns
#     two dictionaries.
#     '''
#     vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding='utf-8').read().splitlines()]
#     token2idx = {token: idx for idx, token in enumerate(vocab)}
#     idx2token = {idx: token for idx, token in enumerate(vocab)}
#     return token2idx, idx2token


def load_data(fpath, data_size):
    ''' Loads the preprocessed data pickle.
    This assumes the following:
    - image data is preprocessed with a VGGNet to (196,512) and this array is flattened with numpy C-order (default)
    - text data is preprocessed with Word2Vec to (34,52) and this array is flattened with numpy C-order (default)

    Returns
    data: dict with key=image_id and value=(image encoded,caption encoded)
    '''

    data = pickle.load(open(fpath, 'rb'))

    if data_size < 1.0:
        # Randomly pick data_size percentage of the dataset
        keys = list(data.keys())
        pick = random.sample(keys, int(data_size * len(keys)))
        data = [data[k] for k in pick]

    return data


# def encode(inp, type, dict):
#     '''Converts string to number. Used for `generator_fn`.
#     inp: 1d byte array.
#     type: "x" (source side) or "y" (target side)
#     dict: token2idx dictionary
#
#     Returns
#     list of numbers
#     '''
#     inp_str = inp.decode("utf-8")
#     if type == "x":
#         tokens = inp_str.split() + ["</s>"]
#     else:
#         tokens = ["<s>"] + inp_str.split() + ["</s>"]
#
#     x = [dict.get(t, dict["<unk>"]) for t in tokens]
#     return x

def generator_fn(fpath, data_size, vgg_shape, embed_shape):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        id: picture_id
        x: array of flattened VGG output (192,512) flattened
    labels: tuple of
        id: picture_id
        y: list of token id of the caption (,max_length)

    '''
    image_caption_dict = load_data(fpath, data_size)

    for value in image_caption_dict.values():
        uid, img, caption = value
        yield uid, img.reshape(vgg_shape), caption.reshape(embed_shape)


def input_fn(fpath, batch_size, data_size, vgg_shape, embed_shape, shuffle):
    '''Batchify data
    image_caption_dict: dict of image_id -> (image_id, img_data, encoded caption)
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        id: picture_id  -- (N,)
        x: array of flattened VGG output (196,512) flattened -- (N,196*512)
    labels: tuple of
        id: picture_id -- (N,)
        y: list of token id of the caption (,max_length) -- (N,max_length)
    '''
    shapes = (
        (), vgg_shape, embed_shape
    )
    types = (
        tf.string, tf.float32, tf.float32
    )

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=([fpath, data_size, vgg_shape, embed_shape])
        # <- arguments for generator_fn. converted to np string arrays
    )

    if shuffle:  # for training
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def get_batch(fpath, batch_size, data_size=1.0, vgg_shape=(196, 512), embed_shape=(34, 52), shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath: path to pickle of dataset
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    images_caption_dict = load_data(fpath, data_size)
    batches = input_fn(fpath, batch_size, data_size, vgg_shape, embed_shape, shuffle)
    num_batches = calc_num_batches(len(images_caption_dict), batch_size)
    return batches, num_batches, len(images_caption_dict)


if __name__ == '__main__':
    # Eager execution
    tf.enable_eager_execution()
    dir_path = os.getcwd()

    # Parse hyperparameters
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    print(hp)

    # Create batches
    train_batches, num_train_batches, num_train_samples = get_batch(hp.dev, hp.batch_size, shuffle=False)

    t, num, train = get_batch(hp.dev, 128, data_size=0.1)

    # Try to mimick the train.py script with eager execution()
    # create a iterator of the correct shape and type
    # iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    # train_init_op = iter.make_initializer(train_batches)

    # id, xs, ys = iter.get_next()

    # Model things
    # print("Loading model")

    # m = EncoderDecoder(hp)

    # loss, train_op, global_step, train_summaries = m.train(xs, ys)
    # y_hat, summaries = m.eval(id, xs, ys)
