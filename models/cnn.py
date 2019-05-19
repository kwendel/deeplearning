# from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.layers import Reshape, Dense
from tensorflow.python.keras import Model

from tensorflow.python.keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import VGG16

from utils.hparams import Hparams


class CNN:
    def __init__(self, hp: Hparams):
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
        model = VGG16()
        print("## VGG unchanged")
        model.summary()

        # Change the model to output his intermediate predictions
        new_input = model.input
        semantic_output = model.layers[-1].output
        hidden_layer = model.layers[-6].output

        print("## VGG with intermediate predictions")
        self.vgg = Model(inputs=new_input, outputs=[hidden_layer, semantic_output])
        self.vgg.summary()

    def predict(self, x):
        return self.vgg.predict(x)


