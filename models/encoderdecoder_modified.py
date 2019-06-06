import tensorflow as tf
from encoder import Encoder
from transformer import Transformer
from utils.data_load import load_vocab
from utils.modules import label_smoothing, noam_scheme
from utils.utils import convert_idx_to_token_tensor

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class EncoderDecoder:

    def __init__(self, hp):
        self.hp = hp
        self.encoder = Encoder(hp)
        self.decoder = Transformer(hp)

        # TODO:
        self.token2idx, self.idx2token = load_vocab(hp.vocab)

    def train(self, xs, ys):
        memory = self.encoder.encode(xs, training=True)
        yhat, y, length = self.decoder.decode(ys, memory, training=True)

        yhat_n = tf.linalg.l2_normalize(yhat, axis = -1)
        y_n = tf.linalg.l2_normalize(y, axis = -1)
        # train scheme 
        loss = tf.losses.cosine_distance(y_n, yhat_n, axis = -1)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory = self.encoder.encode(xs)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            y_hat, y, sents2 = self.decoder.decode(ys, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        # n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        # sent1 = sents1[n]
        # pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        # sent2 = sents2[n]

        # tf.summary.text("sent1", sent1)
        # tf.summary.text("pred", pred)
        # tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries
