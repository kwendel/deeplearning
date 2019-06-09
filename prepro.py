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

import pandas as pd
from numpy.random import seed
from tensorflow import set_random_seed

from preprocess.dataset import load_flickr_set
from preprocess.image import files_to_prediction
from preprocess.text import read_captions, clean_captions
from preprocess.word2vec import create_embeddings
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

    # Check the max length of the the captions
    max_length = captions['caption'].str.split(" ").map(len).max()
    logging.info(f"# Max length of encoded caption = {max_length}")
    logging.critical(f"Set Hyperparameter max length to = {max_length + 2} (maxlength+start/end token)")

    # Check if we have the embeddings of Word2Vec
    if 'word2vec' not in captions.columns:
        captions = create_embeddings(captions, max_length)
        captions.to_pickle(path=caption_pickle)

    # Create the predefined splits with our preprocessed data
    logging.info("# Making data splits with encoded and padded captions")

    def __write_set(path, name, test):
        dataset = load_flickr_set(images, captions, path, test=test)
        p = os.path.join(prepro_path, name)
        pickle.dump(dataset, file=open(p, 'wb'))
        logging.info(f"Succesfully saved in {p}")

    __write_set(dev_path, 'dev_set.pkl', test=False)
    __write_set(train_path, 'train_set.pkl', test=False)
    __write_set(test_path, 'test_set.pkl', test=False)


def test_prepro():
    logging.info("Test if the pickles can be decoded correctly")
    dev_path = os.path.join(prepro_path, 'dev_set.pkl')
    train_path = os.path.join(prepro_path, 'train_set.pkl')
    test_path = os.path.join(prepro_path, 'test_set.pkl')
    dev = pickle.load(open(dev_path, 'rb'))
    trn = pickle.load(open(train_path, 'rb'))
    tst = pickle.load(open(test_path, 'rb'))

    def __print_random(values):
        id, x, ys = rn.choice(values)

        print(f"Picture -- {id}")
        print("Flattened VGG Picture data:")
        print(x)

        # Now do a size check
        vggsize = 196 * 512
        got = len(x)
        print(f"Expected vgg size: {vggsize}, got: {got}")
        if vggsize != got:
            logging.error("VGG predictions are not of the correct size!!")

        print("Flattened Embedded captions:")
        print(ys)

        # Size check
        word2vec_size = 34 * 52
        got = len(ys)
        print(f"Expected embedding size: {word2vec_size}, got: {got}")

        if word2vec_size != got:
            logging.error("Word2Vec Embeddings are not of the correct size!!")

        # if type(ys) is tuple:
        #     # Multiple labels
        #     captions = [sp.DecodeIds(y.tolist()) for y in ys]
        #     print(captions)
        # else:
        #     print(sp.DecodeIds(ys.tolist()))

    print("Dev set")
    __print_random(list(dev.values()))
    print("Train set")
    __print_random(list(trn.values()))
    print("Test set")
    __print_random(list(tst.values()))


if __name__ == '__main__':
    # Parse cmdline arguments
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    # Define directory paths
    dir_path = os.path.join(os.getcwd(), "dataset", "Flickr8k")
    images_path = os.path.join(dir_path, "Flickr8k_Dataset", "Flicker8k_Dataset")
    text_path = os.path.join(dir_path, "Flickr8k_text")
    prepro_path = os.path.join(dir_path, "prepro")

    # Preprocess the data
    setseed()
    prepro(hp)

    # Test the preprocessed files
    test_prepro()
    logging.info("Done")
