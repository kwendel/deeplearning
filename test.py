# -*- coding: utf-8 -*-
# /usr/bin/python3

import logging
import os

import tensorflow as tf

from models.encoderdecoder import EncoderDecoder
from utils.data_load import get_batch
from utils.hparams import Hparams
from utils.utils import get_hypotheses, load_hparams

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples = get_batch(hp.test, hp.test_batch_size, shuffle=False)

iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
id, xs, ys = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = EncoderDecoder(hp)
y_hat, _ = m.eval(id, xs, ys)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_  # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)
    sess.run(test_init_op)

    logging.info("# get hypotheses")
    hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.vec2word)

    logging.info("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir):
        os.makedirs(hp.testdir)

    captions = os.path.join(hp.testdir, model_output)
    with open(captions, 'w') as fout:
        fout.write("\n".join(hypotheses))
