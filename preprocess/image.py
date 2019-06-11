import logging
import os
from collections import OrderedDict

from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

from models.cnn import DataPredictor


def files_to_prediction(dir_path):
    """ Returns the VGG16 intermediate predictions for each .jpg picture in the directory

    :param dir_path: directory path to the pictures
    :return: ordereddict with key=id and value=predictions
    """
    dp = DataPredictor()
    images = OrderedDict()
    pixels = 224  # size required for VGG
    t_size = (pixels, pixels, 3)  # RGB requires 3 channels

    files = os.listdir(dir_path)
    for i, fname in enumerate(tqdm(files, desc='Making VGG intermediate prediction')):
        if not fname.endswith(".jpg"):
            logging.warning("Encountered file that was not a .jpg in dataset, ignore for now -- {}".format(fname))
            continue

        # Read image
        path = os.path.join(dir_path, fname)

        img = load_img(path, target_size=t_size)

        # To numpy data
        img = img_to_array(img)
        # Preprocess with keras as it requires to be preprocess just like VGG expects
        img = preprocess_input(img)

        # Reshape to one row
        img = img.reshape((1,) + img.shape[:3])

        images[fname] = dp.predict(img).flatten()

    return images
