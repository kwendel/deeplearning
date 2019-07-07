# -*- coding: utf-8 -*-
# /usr/bin/python3
import logging
import math
import os

import tensorflow as tf
from tqdm import tqdm

from models.encoderdecoder import EncoderDecoder
from utils.data_load import get_batch
from utils.hparams import Hparams
from utils.utils import save_hparams, save_variable_specs

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")

# Overfit test case -- train on a small subset of the development set to see if the loss converges
train_batches, num_train_batches, num_train_samples = get_batch(hp.minidev, hp.batch_size, shuffle=True)

# Normal case -- train on the shuffled train set but evaluated on the unshuffeled development set
# train_batches, num_train_batches, num_train_samples = get_batch(hp.train, hp.batch_size,
#                                                              data_size=hp.split_size, shuffle=True)

# Evaluation set
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.minidev, hp.eval_batch_size,
                                                             data_size=hp.split_size, shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
id, xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = EncoderDecoder(hp)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat, eval_summaries = m.eval(id, xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)

with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps + 1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss)  # train loss

            logging.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# write results")
            model_output = "flickr8k_E%02dL%.2f" % (epoch, _loss)

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()

logging.info("Done")
