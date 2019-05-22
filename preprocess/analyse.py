# Analysis inspired by
# https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html
import logging
import os
from collections import OrderedDict, Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.python.keras.preprocessing.image import load_img

from .image import files_to_prediction
from .text import read_captions, clean_captions, make_df_word


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


def image_analysis(root, images_dir):
    # Use VGG to make prediction for a whole directory
    preds = files_to_prediction(f"{root}{images_dir}")

    # Analyze the VGG embedding with the help of PCA
    analyse_pca(preds)


def text_analysis(root, text_dir):
    caption_path = f"{root}{text_dir}/Flickr8k.lemma.token.txt"
    topn = 50

    df_txt = read_captions(caption_path)
    df_word = make_df_word(df_txt)

    print("Start raw text analysis")
    print("=" * 30)

    # Find unique image_idx
    uni_image_idx = np.unique(df_txt.image_idx.values)
    print("The number of unique file names : {}".format(len(uni_image_idx)))
    print("The distribution of the number of captions for each image:")
    print(Counter(Counter(df_txt.image_idx.values).values()))
    print()

    # Word analysis
    print("Most occuring words\n")
    print(df_word.head(5))

    plot_hist(df_word.iloc[:topn, :], title="The top " + str(topn) + " most frequently appearing words")
    plot_hist(df_word.iloc[-topn:, :], title="The least " + str(topn) + " most frequently appearing words")

    print("Start cleaned text analysis")
    print("=" * 30)

    df_txt = clean_captions(df_txt)
    df_word = make_df_word(df_txt)
    plot_hist(df_word.iloc[:topn, :], title="The top " + str(topn) + " most frequently appearing words")
    plot_hist(df_word.iloc[-topn:, :], title="The least " + str(topn) + " most frequently appearing words")


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Retrieved from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def plot_hist(dfsub, title="Give a title next time"):
    plt.figure(figsize=(20, 3))
    plt.bar(dfsub.index, dfsub["count"])
    plt.yticks(fontsize=20)
    plt.xticks(dfsub.index, dfsub["word"], rotation=90, fontsize=20)
    plt.title(title, fontsize=20)
    plt.show()


if __name__ == '__main__':
    # TODO: these paths only work on windows, change with os.path
    # Also, fill in your local path to the correct project root
    root = "C:/Users/TODO"
    images_dir = 'dataset/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset'
    text_dir = 'dataset/Flickr8k/Flickr8k_text/'

    logging.warning(f"Using project root -- {root}")

    image_analysis(root, images_dir)
    text_analysis(root, text_dir)
