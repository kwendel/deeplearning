from os import path

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Reshape, Dense
from tensorflow.python.keras.models import Model

from utils.hparams import Hparams

PRETRAINED_VGG16 = 'pretrained/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


class CNN:
    def __init__(self, hp):
        self.hp = hp

        # TODO: why is the Dense output size (*,1000)?
        # I think this should be d_model, the parameter that sets the size of the hidden state

        # TODO/IDEA: Can we simplify this model?
        # For each picture, we can make predictions with VGG and save them -> (#pictures, 4096) (or whatever size VGG outputs in the final layer)
        # Now use these predictions as input data. Now we only need to do Relu + FF for an embedding

        # Construct the CNN model by deleting the last layer and add Relu + FF
        model = VGG16()
        new_input = model.input
        semantic_output = model.layers[-1].output
        hidden_layer = model.layers[-6].output
        hidden_layer = Reshape((196, 512), input_shape=(14, 14, 512))(hidden_layer)
        spatial_output = Dense(1000, activation='relu', input_shape=(196, 512))(hidden_layer)
        # new_output = Reshape((196, 512), input_shape = (14, 14, 512))(hidden_layer)

        self.vgg = Model(inputs=new_input, outputs=[spatial_output, semantic_output])
        self.vgg.summary()

    def encode(self, xs):
        x, seqlens, sents1 = xs
        y_hat = self.vgg.predict(x)

        return y_hat, sents1


class DataPredictor:
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
