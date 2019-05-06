from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np
# load the model
model = VGG16()

# test for coco examples
test = ['data/mug.jpg', 'data/calling.jpg', 'data/elephants.jpg', 'data/football.jpg', 'data/jet.jpg', 'data/kitchen.jpg', 'data/room.jpg', 'data/tennis.jpg', 'data/train.jpg']

for file in test:
    # load an image from file
    image = load_img(file, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # print the classification
    print('test: %s' % file)
    for i in np.arange(5):
        # retrieve the most likely result, e.g. highest probability
        labeli = label[0][i]
        print('%s (%.2f%%)' % (labeli[1], labeli[2]*100))
