# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the iwslt 2016 datasets.
'''

import errno
import logging
import os
import pickle

import numpy as np
import pandas as pd

from data_analyse.image import files_to_prediction, load_flickr_set, get_caption_set
from data_analyse.text import read_captions, clean_captions, train_sp, load_sp, encode_caption_as_ids, add_padding
from utils.hparams import Hparams

logging.basicConfig(level=logging.INFO)


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
        pickle.dump(pad_caps, file=open(enc_path, "wb"))

    # Create the predefined splits with our preprocessed data
    logging.info("# Making data splits with encoded and padded captions")
    dev = load_flickr_set(images, pad_caps, dev_path, test=False)
    train = load_flickr_set(images, pad_caps, train_path, test=False)
    test = load_flickr_set(images, pad_caps, test_path, test=True)

    logging.info("# Writing data pickles")

    def __to_pickle(structure, path):
        p = os.path.join(prepro_path, path)
        pickle.dump(structure, file=open(p, 'wb'))

    __to_pickle(dev, 'dev_set.pkl')
    __to_pickle(train, 'train_set.pkl')
    __to_pickle(test, 'test_set.pkl')

    # max_length = enc_caps.caption.map(len).max()

    # dev = load_flickr_set(images, enc_caps, dev_path, test=False)
    # train = load_flickr_set(images, enc_caps, train_path, test=False)
    # test = load_flickr_set(images, enc_caps, test_path, test=True)

    # # train
    # _prepro = lambda x: [line.strip() for line in open(x, 'r', encoding='utf-8').read().split("\n") \
    #                      if not line.startswith("<")]
    # prepro_train1, prepro_train2 = _prepro(train1), _prepro(train2)
    # assert len(prepro_train1) == len(prepro_train2), "Check if train source and target files match."
    #
    # # eval
    # _prepro = lambda x: [re.sub("<[^>]+>", "", line).strip() \
    #                      for line in open(x, 'r', encoding='utf-8').read().split("\n") \
    #                      if line.startswith("<seg id")]
    # prepro_eval1, prepro_eval2 = _prepro(eval1), _prepro(eval2)
    # assert len(prepro_eval1) == len(prepro_eval2), "Check if eval source and target files match."
    # # test
    # prepro_test1, prepro_test2 = _prepro(test1), _prepro(test2)
    # assert len(prepro_test1) == len(prepro_test2), "Check if test source and target files match."
    #
    # logging.info("Let's see how preprocessed data look like")
    # logging.info("prepro_train1:", prepro_train1[0])
    # logging.info("prepro_train2:", prepro_train2[0])
    # logging.info("prepro_eval1:", prepro_eval1[0])
    # logging.info("prepro_eval2:", prepro_eval2[0])
    # logging.info("prepro_test1:", prepro_test1[0])
    # logging.info("prepro_test2:", prepro_test2[0])
    #
    # logging.info("# write preprocessed files to disk")
    # os.makedirs("iwslt2016/prepro", exist_ok=True)
    #
    # def _write(sents, fname):
    #     with open(fname, 'w', encoding='utf-8') as fout:
    #         fout.write("\n".join(sents))
    #
    # _write(prepro_train1, "iwslt2016/prepro/train.de")
    # _write(prepro_train2, "iwslt2016/prepro/train.en")
    # _write(prepro_train1 + prepro_train2, "iwslt2016/prepro/train")
    # _write(prepro_eval1, "iwslt2016/prepro/eval.de")
    # _write(prepro_eval2, "iwslt2016/prepro/eval.en")
    # _write(prepro_test1, "iwslt2016/prepro/test.de")
    # _write(prepro_test2, "iwslt2016/prepro/test.en")
    #
    # logging.info("# Train a joint BPE model with sentencepiece")
    # os.makedirs("iwslt2016/segmented", exist_ok=True)
    # train = '--input=iwslt2016/prepro/train --pad_id=0 --unk_id=1 \
    #          --bos_id=2 --eos_id=3\
    #          --model_prefix=iwslt2016/segmented/bpe --vocab_size={} \
    #          --model_type=bpe'.format(hp.vocab_size)
    # spm.SentencePieceTrainer.Train(train)
    #
    # logging.info("# Load trained bpe model")
    # sp = spm.SentencePieceProcessor()
    # sp.Load("iwslt2016/segmented/bpe.model")
    #
    # logging.info("# Segment")
    #
    # def _segment_and_write(sents, fname):
    #     with open(fname, "w", encoding='utf-8') as fout:
    #         for sent in sents:
    #             pieces = sp.EncodeAsPieces(sent)
    #             fout.write(" ".join(pieces) + "\n")
    #
    # _segment_and_write(prepro_train1, "iwslt2016/segmented/train.de.bpe")
    # _segment_and_write(prepro_train2, "iwslt2016/segmented/train.en.bpe")
    # _segment_and_write(prepro_eval1, "iwslt2016/segmented/eval.de.bpe")
    # _segment_and_write(prepro_eval2, "iwslt2016/segmented/eval.en.bpe")
    # _segment_and_write(prepro_test1, "iwslt2016/segmented/test.de.bpe")
    #
    # logging.info("Let's see how segmented data look like")
    # print("train1:", open("iwslt2016/segmented/train.de.bpe", 'r', encoding='utf-8').readline())
    # print("train2:", open("iwslt2016/segmented/train.en.bpe", 'r', encoding='utf-8').readline())
    # print("eval1:", open("iwslt2016/segmented/eval.de.bpe", 'r', encoding='utf-8').readline())
    # print("eval2:", open("iwslt2016/segmented/eval.en.bpe", 'r', encoding='utf-8').readline())
    # print("test1:", open("iwslt2016/segmented/test.de.bpe", 'r', encoding='utf-8').readline())


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    # prepro(hp)
    dir_path = os.path.join(os.getcwd(), "dataset", "Flickr8k")
    images_path = os.path.join(dir_path, "Flickr8k_Dataset", "Flicker8k_Dataset")
    text_path = os.path.join(dir_path, "Flickr8k_text")
    prepro_path = os.path.join(dir_path, "prepro")

    # logging.info("Done")
