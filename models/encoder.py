import tensorflow as tf
from utils.modules import get_token_embeddings, ff, positional_encoding, multihead_attention
import logging

logging.basicConfig(level=logging.INFO)


class Encoder:
    def __init__(self, hp):
        self.hp = hp

    def encode(self, xs, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):

            sampleId, image = xs
            enc =  tf.get_variable('image', image, dtype=tf.float32)
            enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model], scope='encoder')

        return enc
