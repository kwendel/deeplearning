import tensorflow as tf
from transformer import Transfomer
import numpy as np
from utils.hparams import Hparams

Hp = Hparams()
parser = Hp.parser
hp = parser.parse_args()

hp.vocab_size = 40
hp.d_model = 50


print(hp)
print(hp.d_model)

memory = np.random.rand(512, hp.d_model)
transformer = Transfomer(hp)


