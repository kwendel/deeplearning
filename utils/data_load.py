# -*- coding: utf-8 -*-
# /usr/bin/python3
import pickle
import random

import tensorflow as tf

from utils.utils import calc_num_batches


def load_data(fpath, data_size):
    """ Loads the preprocessed data pickle.
    This assumes the following:
    - image data is preprocessed with a VGGNet to (196,512) and this array is flattened with numpy C-order (default)
    - text data is preprocessed with Word2Vec to (34,52) and this array is flattened with numpy C-order (default)

    Params
    data_size: percentage of the dataset that is randomly picked

    Returns
    data: dict with key=image_id and value=(id, image encoded,caption encoded)
    """

    data = pickle.load(open(fpath, 'rb'))

    if data_size < 1.0:
        # Randomly pick data_size percentage of the dataset
        keys = list(data.keys())
        pick = random.sample(keys, int(data_size * len(keys)))
        data = {k: data[k] for k in pick}

    return data


def generator_fn(fpath, data_size, vgg_shape, embed_shape):
    """ Generates training / evaluation data
    fpath: path to data pickle
    data_size: percentage of dataset used
    vgg_shape: shape to reshape vgg data to
    embed_shape: shape to reshape caption data to

    yields
    (id, xs, ys)
    id = picture_id
    xs = array of flattened VGG output (192,512) flattened
    ys = list of token id of the caption (,max_length)
    """
    image_caption_dict = load_data(fpath, data_size)

    for value in image_caption_dict.values():
        uid, img, caption = value
        yield uid, img.reshape(vgg_shape), caption.reshape(embed_shape)


def input_fn(fpath, batch_size, data_size, vgg_shape, embed_shape, shuffle):
    """ Batchify data
    image_caption_dict: dict of image_id -> (image_id, img_data, encoded caption)
    batch_size: scalar
    data_size: scalar
    vgg_shape: tuple
    embed_shape: tuple
    shuffle: boolean

    Returns
    tf.Dataset with (id, xs, ys)
    id: (N,)
    xs: (N,196,512)
    ys: (N,34,52)
    """
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
    """Gets training / evaluation mini-batches
    fpath: path to pickle of dataset
    batch_size: scalar
    data_size: scalar between 0 and 1
    vgg_shape: tuple
    embed_shape: tuple
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    """
    images_caption_dict = load_data(fpath, data_size)
    batches = input_fn(fpath, batch_size, data_size, vgg_shape, embed_shape, shuffle)
    num_batches = calc_num_batches(len(images_caption_dict), batch_size)
    return batches, num_batches, len(images_caption_dict)
