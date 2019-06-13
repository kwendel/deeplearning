import tensorflow as tf
from utils.modules import get_token_embeddings, ff, positional_encoding, multihead_attention
import logging

logging.basicConfig(level=logging.INFO)


class Transformer:
    def __init__(self, hp):
        self.hp = hp
        # self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def decode(self, ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model). float32.

        Returns
        y_hat: (N, T2, V). float32.
        y: (N, T2, V). float32.
        sents2: (N,). string. (staat deze er nog in?)
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # Word2Vec embedding
            decoder_inputs = ys

            # embedding
            dec = decoder_inputs  # (N, T2, V)
            # dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff_trans, self.hp.d_model_trans])

        
        y_hat = dec

        return y_hat
