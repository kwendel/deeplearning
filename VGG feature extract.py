import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Dense
import numpy as np
# load the model
model = VGG16()
model.summary()

# make new model by deleting last layer
new_input = model.input
semantic_output = model.layers[-1].output
hidden_layer = model.layers[-6].output
hidden_layer = Reshape((196, 512), input_shape = (14, 14, 512))(hidden_layer)
spatial_output = Dense(1000, activation = 'relu', input_shape = (196,512)) (hidden_layer)
#new_output = Reshape((196, 512), input_shape = (14, 14, 512))(hidden_layer)

image_features_extract_model = Model(inputs = new_input, outputs = [spatial_output, semantic_output])
image_features_extract_model.summary()


# test for coco examples
test = 'data/mug.jpg'


# load an image from file
image = load_img(test, target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = image_features_extract_model.predict(image)
