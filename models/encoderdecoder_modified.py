import logging

import tensorflow as tf
from tqdm import tqdm

from encoder import Encoder
from preprocess.word2vec import START_VEC, END_VEC, UNK_VEC, PAD_VEC
from transformer_modified import Transformer
from utils.modules import noam_scheme

logging.basicConfig(level=logging.INFO)


class EncoderDecoder:

    def __init__(self, hp):
        self.hp = hp
        self.encoder = Encoder(hp)
        self.decoder = Transformer(hp)

        self.embedding = {
            'start': tf.convert_to_tensor(START_VEC, dtype=tf.float32),
            'end': tf.convert_to_tensor(END_VEC, dtype=tf.float32),
            'unk': tf.convert_to_tensor(UNK_VEC, dtype=tf.float32),
            'pad': tf.convert_to_tensor(PAD_VEC, dtype=tf.float32)
        }

    def train(self, xs, ys):
        memory = self.encoder.encode(xs)
        yhat = self.decoder.decode(ys, memory, training=True)

        yhat_n = tf.linalg.l2_normalize(yhat, axis=-1)
        y_n = tf.linalg.l2_normalize(ys, axis=-1)
        # train scheme
        loss = tf.losses.cosine_distance(y_n, yhat_n, axis=-1)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, id, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2, V)
        '''

        # Initialize batch (N, 1, 52) with only first row with the start token
        y_start = tf.ones((tf.shape(xs)[0], 1, 52), tf.float32) * self.embedding['start']
        y_in = y_start

        # Use Encoder to generate memory of the picture
        memory = self.encoder.encode(xs)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            y_hat = self.decoder.decode(y_in, memory, training=False)

            # TODO: maybe this is to strict? We can also do something like tf.math.less_equal(abs(rows - pad), epsi)...
            # Check if the last row is completely/element-wise the same as the pad token
            equals = tf.math.equal(y_hat[:, -1, :], self.embedding['pad'])

            # Check for all batches at the same time if this is the case
            if tf.reduce_all(equals):
                # Then we can stop evaluating
                break

            # Grow the input to the decoder with one row
            y_in = tf.concat((y_start, y_hat), 1)

        # TODO: monitor a random sample
        # true value is ys, last prediction is y_hat

        # n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        # sent1 = sents1[n]
        # pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        # sent2 = sents2[n]

        # tf.summary.text("sent1", sent1)
        # tf.summary.text("pred", pred)
        # tf.summary.text("sent2", sent2)
        # summaries = tf.summary.merge_all()
        summaries = None

        return y_hat, summaries
