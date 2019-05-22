# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
May 2019 by Kasper Wendel

Preprocess the Flickr8k dataset
Images -> VGG intermediate predictions
captions -> encoded with Tokenizer SentencePiece as indices
'''

import errno
import logging
import os
import pickle
import random as rn

import numpy as np
import pandas as pd
from numpy.random import seed
from tensorflow import set_random_seed

from preprocess.dataset import load_flickr_set, get_caption_set
from preprocess.image import files_to_prediction
from preprocess.text import read_captions, clean_captions, train_sp, load_sp, encode_caption_as_ids, add_padding
from utils.hparams import Hparams

logging.basicConfig(level=logging.INFO)


def setseed(sd=42):
    seed(sd)
    set_random_seed(sd)
    rn.seed(sd)


def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """

    # Define directory paths
    dir_path = os.path.join(os.getcwd(), "dataset", "Flickr8k")
    images_path = os.path.join(dir_path, "Flickr8k_Dataset", "Flicker8k_Dataset")
    text_path = os.path.join(dir_path, "Flickr8k_text")
    prepro_path = os.path.join(dir_path, "prepro")

    # Check directory paths
    logging.info("# Using directories")
    logging.info(f"Dataset directory -- {dir_path}")
    logging.info(f"Images directory -- {images_path}")
    logging.info(f"Text directory -- {text_path}")
    logging.info(f"Preprocessed saved in directory -- {prepro_path}")
    logging.info("# Check if the dataset directories exist")
    for d in (dir_path, images_path, text_path):
        if not os.path.isdir(d):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), d)

    # Create directory for saving preprocessed data
    logging.info("# Create directory for preprocessed data")
    os.makedirs(prepro_path, exist_ok=True)

    # Check dataset splits files exist
    logging.info("# Check if dataset files are existing")
    dev_path = os.path.join(text_path, 'Flickr_8k.devImages.txt')
    train_path = os.path.join(text_path, 'Flickr_8k.trainImages.txt')
    test_path = os.path.join(text_path, 'Flickr_8k.testImages.txt')
    for f in (dev_path, train_path, test_path):
        if not os.path.exists(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    # Preprocess images -- make (196, 512) prediction per image with VGG
    logging.info("# Preprocessing images")
    images_pickle = os.path.join(prepro_path, "images.pkl")
    if os.path.exists(images_pickle):
        logging.info("# Loading images pickle from -- {}".format(images_pickle))
        images = pickle.load(open(images_pickle, 'rb'))
    else:
        images = files_to_prediction(images_path)
        logging.info("# Image pickle saved in -- {}".format(images_pickle))
        pickle.dump(images, file=open(images_pickle, "wb"))

    # Preprocess captions -- clean captions (remove punct, single chars and numbers)
    logging.info("# Preprocessing captions")
    caption_pickle = os.path.join(prepro_path, "cleaned_captions.pkl")
    if os.path.exists(caption_pickle):
        logging.info("# Loading caption pickle from -- {}".format(caption_pickle))
        captions = pd.read_pickle(caption_pickle)
    else:
        captions = read_captions(os.path.join(text_path, 'Flickr8k.lemma.token.txt'))
        captions = clean_captions(captions)
        logging.info("# Caption pickle saved in -- {}".format(caption_pickle))
        captions.to_pickle(path=caption_pickle)

    # Train the SentencePiece model on the training captions
    logging.info("# Checking SP model")
    model_prefix = os.path.join('dataset', 'Flickr8k', 'prepro', 'trained_sp')
    model_path = os.path.join(prepro_path, 'trained_sp.model')
    vocab_path = os.path.join(prepro_path, 'trained_sp.vocab')
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        logging.info("# Load SP model from -- {}".format(model_path))
        # Load the sp model
        sp = load_sp(model_path)
    else:
        # IMPORTANT: only train the sp on the training samples!!
        logging.info("# Gather training captions")
        trn_captions = get_caption_set(captions, train_path)

        # Save the captions as numpy array
        trn_captions_path = os.path.join('dataset', 'Flickr8k', 'prepro', 'train_captions.txt')
        np.savetxt(trn_captions_path, np.array(trn_captions), fmt='%s')

        # Train the SP model with the training captions
        logging.info("# Train SP with training captions")
        train_sp(trn_captions_path, model_prefix, hp.vocab_size)
        sp = load_sp(model_path)

    # Encode the captions as ids with SP
    enc_path = os.path.join(prepro_path, 'enc_caps')
    pad_path = os.path.join(prepro_path, 'pad_caps')
    if os.path.exists(enc_path):
        enc_caps = pickle.load(open(enc_path, 'rb'))
    else:
        enc_caps = encode_caption_as_ids(captions, sp)
        pickle.dump(enc_caps, file=open(enc_path, "wb"))

    # Check the max length of the encoded captions
    max_length = enc_caps['caption'].map(len).max()
    logging.info(f"# Max length of encoded caption = {max_length}")

    # Pad senteces to equal length
    if os.path.exists(pad_path):
        pad_caps = pickle.load(open(pad_path, 'rb'))
    else:
        pad_caps = add_padding(enc_caps, max_length)
        pickle.dump(pad_caps, file=open(pad_path, "wb"))

    # Create the predefined splits with our preprocessed data
    logging.info("# Making data splits with encoded and padded captions")

    def __write_set(path, name, test):
        dataset = load_flickr_set(images, pad_caps, path, test=test)
        p = os.path.join(prepro_path, name)
        pickle.dump(dataset, file=open(p, 'wb'))
        logging.info(f"Succesfully saved in {p}")

    __write_set(dev_path, 'dev_set.pkl', test=False)
    __write_set(train_path, 'train_set.pkl', test=False)
    __write_set(test_path, 'test_set.pkl', test=True)


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    setseed()
    prepro(hp)
    logging.info("Done")
