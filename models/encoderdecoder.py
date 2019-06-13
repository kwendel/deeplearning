import logging

import tensorflow as tf

from preprocess.vec2word import Vec2Word
from preprocess.word2vec import START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN
from utils.modules import noam_scheme
from utils.utils import convert_embedding_tensor
from .encoder import Encoder
from .transformer import Transformer

logging.basicConfig(level=logging.INFO)


class EncoderDecoder:

    def __init__(self, hp):
        self.hp = hp
        self.encoder = Encoder(hp)
        self.decoder = Transformer(hp)

        self.vec2word = Vec2Word.load_model(hp.vec2word, hp.embed_size)

        self.embedding = {
            START_TOKEN: tf.constant(self.vec2word.get_vec(START_TOKEN), dtype=tf.float32),
            END_TOKEN: tf.constant(self.vec2word.get_vec(END_TOKEN), dtype=tf.float32),
            PAD_TOKEN: tf.constant(self.vec2word.get_vec(PAD_TOKEN), dtype=tf.float32),
            UNK_TOKEN: tf.constant(self.vec2word.get_vec(UNK_TOKEN), dtype=tf.float32)
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

    def eval(self, ids, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2, V)
        '''

        # Initialize batch (N, 1, 52) with only first row with the start token
        y_start = tf.ones((tf.shape(xs)[0], 1, self.hp.embed_size), tf.float32) * self.embedding[START_TOKEN]
        # tf.print(y_start[0])
        y_hat = y_start
        # tf.print(y_hat[0])

        # Use Encoder to generate memory of the picture
        memory = self.encoder.encode(xs)

        logging.info("Inference graph is being built. Please be patient.")

        maxlen = tf.constant(self.hp.maxlen2)

        def cond(y_in, c):
            notLast = tf.math.less(c, maxlen)

            # Check if the last row is equal to the pad token
            # TODO: maybe this is to strict? We can also do something like tf.math.less_equal(abs(rows - pad), epsi)...
            equals = tf.math.equal(y_in[:, -1, :], self.embedding[PAD_TOKEN])
            notPad = tf.logical_not(tf.reduce_all(equals))

            return tf.logical_and(notLast, notPad)

        def body(y_in, c):
            y_hat_ = self.decoder.decode(y_in, memory, training=False)

            # Grow the input to the decoder with one row
            y_in = tf.concat((y_start, y_hat_), 1)
            c = tf.add(c, 1)

            return y_in, c

        c = tf.constant(0)
        # c = 0
        y_hat = tf.while_loop(
            cond,
            body,
            [y_hat, c],
            # Let tf know that y_in is growing each iteration
            shape_invariants=[tf.TensorShape([None, None, self.hp.embed_size]), c.get_shape()]
        )

        # tf.print(y_hat[0])
        # tf.print(tf.shape(y_hat))


        # Monitor a random samples
        # true value is ys, last prediction is y_hat

        n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        id = ids[n]

        # Convert back to the sentences
        real_scores, real_sent = convert_embedding_tensor(ys[n], self.vec2word)
        pred_scores, pred_sent = convert_embedding_tensor(y_hat[n], self.vec2word)

        # for s in (real_scores, real_sent, pred_scores, pred_sent):
        #     tf.print(s)

        # Save summary
        tf.summary.text("id", id)
        tf.summary.text("pred_scores", pred_scores)
        tf.summary.text("pred_sent", pred_sent)
        tf.summary.text("real_scores", real_scores)
        tf.summary.text("real_sent", real_sent)
        summaries = tf.summary.merge_all()
        # summaries = None

        return y_hat, summaries
