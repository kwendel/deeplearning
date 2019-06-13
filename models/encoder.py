import logging

from utils.modules import ff

logging.basicConfig(level=logging.INFO)


class Encoder:
    """
    Encoder: this class is used for applying a fully connected Feedforward layer with ReLu activation on the VGG
    predictions. T
    """

    def __init__(self, hp):
        self.hp = hp

    def encode(self, xs):
        """
        xs: image data (N,T,C) -> (batch_size, 196,512)

        enc: encoded image data -> (batch_size,??)

        """
        enc = ff(xs, num_units=[self.hp.d_ff_enc, self.hp.d_model_enc], scope='vgg-encoder', residual=False)

        return enc
