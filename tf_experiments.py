import tensorflow as tf

from utils.data_load import get_batch
from utils.hparams import Hparams

if __name__ == '__main__':
    # Eager execution
    tf.enable_eager_execution()

    # Parse hyperparameters
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    print(hp)

    # Create batches
    # train_batches, num_train_batches, num_train_samples = get_batch(hp.dev, hp.batch_size, shuffle=False)
    train_batches, num_train_batches, num_train_samples = get_batch(hp.dev, hp.batch_size, data_size=0.1, shuffle=False)

    # Call the batches to construct them and see what they contain
    for val in train_batches.take(1):
        tf.print(val)
        print(val)

    # Try to mimick the train.py script with eager execution()
    # create a iterator of the correct shape and type
    # iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    # train_init_op = iter.make_initializer(train_batches)

    # id, xs, ys = iter.get_next()

    # Model things
    # print("Loading model")

    # m = EncoderDecoder(hp)

    # loss, train_op, global_step, train_summaries = m.train(xs, ys)
    # y_hat, summaries = m.eval(id, xs, ys)
