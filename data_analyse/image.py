import os
from collections import OrderedDict

from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing.image import load_img, img_to_array

from models.cnn import DataPredictor

images_dir = 'dataset/Flicker8k/Flicker8k_Dataset/Flicker8k_Dataset'
text_dir = 'dataset/Flicker8k/Flicker8k_text/'


def files_to_prediction(dir_path):
    images = OrderedDict()
    pixels = 256
    t_size = (pixels, pixels, 3)  # RGB requires 3 channels
    # vgg = cnn.

    files = os.listdir(dir_path)
    for i, fname in enumerate(files):
        # Read image
        path = f"{dir_path}/{fname}"
        img = load_img(path, target_size=t_size)

        # To numpy data
        img = img_to_array(img)
        # Preprocess with keras as it requires to be preprocess just like VGG expects
        img = preprocess_input(img)

        # images[fname] =


if __name__ == '__main__':
    dp = DataPredictor()

    # images = OrderedDict()
