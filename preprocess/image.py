import logging
import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

from models.cnn import DataPredictor

root = "C:/Users/kaspe/Documents/Study/Q3 Deep Learning/deeplearning/"
images_dir = 'dataset/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset'
text_dir = 'dataset/Flickr8k/Flickr8k_text/'


def files_to_prediction(dir_path):
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

    # Save the results
    # pickle.dump(images, file=open(f"{root}{dir_path}/preprocessed.p", "wb"))

    return images


def analyse_pca(images: dict):
    # Some randomly selected pictures that are close to each other to analyze the clusters
    picked_pic = OrderedDict()
    picked_pic["red"] = [7507, 6065, 3207, 7742, 3434]
    picked_pic["green"] = [2173, 6017, 7420, 1345, 691]
    picked_pic["magenta"] = [827, 2210, 379, 686, 2210]
    picked_pic["blue"] = [2098, 7927, 6566, 6502]
    picked_pic["yellow"] = [5410, 1357, 4074]
    picked_pic["purple"] = [3793]

    # Map to 2D for visualization
    encoded_data = np.array(list(images.values()))
    pca = PCA(n_components=2, random_state=123)
    y_pca = pca.fit_transform(encoded_data)

    # Plot PCA embedding
    plt.figure(figsize=(15, 15), dpi=100)
    plt.scatter(y_pca[:, 0], y_pca[:, 1], c='white')
    for irow in range(y_pca.shape[0]):
        plt.annotate(irow, y_pca[irow, :], color="black", alpha=0.5)
    for color, irows in picked_pic.items():
        for irow in irows:
            plt.annotate(irow, y_pca[irow, :], color=color)
    plt.xlabel("pca embedding 1", fontsize=30)
    plt.ylabel("pca embedding 2", fontsize=30)
    plt.show()

    # Plot images
    fig = plt.figure(figsize=(16, 20))
    count = 1
    pixels = 224  # size required for VGG
    t_size = (pixels, pixels, 3)  # RGB requires 3 channels

    jpgs = os.listdir(os.path.join(root, images_dir))
    for color, irows in picked_pic.items():
        for ivec in irows:
            name = jpgs[ivec]
            filename = f"{root}{images_dir}/{name}"
            image = load_img(filename, target_size=t_size)
            ax = fig.add_subplot(len(picked_pic), 5, count,
                                 xticks=[], yticks=[])
            count += 1
            plt.imshow(image)
            plt.title("{} ({})".format(ivec, color))
    plt.show()


def load_flickr_train(images, text, ):
    return load_flickr_set(images, text, 'Flickr_8k.trainImages.txt', test=False)


def load_flickr_test(images, text):
    return load_flickr_set(images, text, 'Flickr_8k.testImages.txt', test=True)


def load_flickr_dev(images, text):
    return load_flickr_set(images, text, 'Flickr_8k.testImages.txt', test=True)


def load_flickr_set(images, text, file, test):
    dataset = OrderedDict()
    set_ids = read_ids(file)

    for pic_id in tqdm(set_ids, desc="Generating Flickr dataset"):
        img_data = images[pic_id]
        text_data = text[text['image_idx'] == pic_id]

        # For testing data, we want all available captions as true labels
        # result -> (id, x, (y1, y2, .., y5))
        if test:
            labels = ()
            for idx, row in text_data.iterrows():
                labels += (row['caption'],)

            dataset[pic_id] = (pic_id, img_data, labels)
        # For training data, we want to use the captions to generate new training objects
        # result -> (id, x, y)
        else:
            for idx, row in text_data.iterrows():
                new_name = pic_id + '-' + row['caption_idx']
                dataset[new_name] = (new_name, img_data, row['caption'])

    return dataset


def read_ids(file):
    set_ids = []
    with open(file, 'r', encoding='UTF-8') as f:
        # Read the data
        for line in f:
            # Strip the ending newline char
            set_ids.append(line.strip("\n"))

    return set_ids


def get_caption_set(text, file):
    set_ids = read_ids(file)
    result = []

    for pic_id in tqdm(set_ids, desc="Generating captions training set"):
        text_data = text[text['image_idx'] == pic_id]

        for _, row in text_data.iterrows():
            result.append(row['caption'])

    return result


if __name__ == '__main__':
    # Use VGG to make prediction for a whole directory
    # files_to_prediction(images_dir)

    text = pd.read_pickle('encoded_captions.p')
    preds = pickle.load(open(f"{root}{images_dir}/preprocessed.p", "rb"))
    trn = load_flickr_train(preds, text)
    tst = load_flickr_test(preds, text)

    # Analyze the VGG embedding with the help of PCA
    # analyse_pca(preds)