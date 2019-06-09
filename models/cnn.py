from os import path

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.models import Model

PRETRAINED_VGG16 = 'pretrained/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


class DataPredictor:
    """
    DataPredictor is used as a preprocess step to speed up network training.
    It uses a VGG16 with pretrained ImageNet weights to gather the intermediate predictions
    """

    def __init__(self):
        # Construct the pretrained VGG16
        dir = path.dirname(__file__)
        weights_path = path.join(dir, PRETRAINED_VGG16)
        model = VGG16(include_top=True, weights=weights_path)

        # Output the original network structure for debuggin
        # print("## VGG unchanged")
        # model.summary()

        # Change the model to output his intermediate predictions
        new_input = model.input
        new_output = model.layers[-6].output
        reshaper = Reshape((196, 512), input_shape=(14, 14, 512))(new_output)

        # Check if architecture of the VGG with intermediate predictions
        print("## VGG with intermediate predictions")
        model = Model(inputs=new_input, outputs=reshaper)

        self.vgg = model

    def predict(self, x):
        return self.vgg.predict(x)
