# Inspired of https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import pandas as pd
from collections import Counter
import logging

captions_path = "C:/Users/User/Downloads/Flickr8k_text/Flickr8k.lemma.token.txt"


def read_file(path):
    """"Reads the file specified in path"""
    file = open(path, 'r')
    text = file.read()
    file.close()
    return text


def split_lines(text):
    """"Splits each line in the specified text into different cols."""
    datatxt = []

    # For each line in the file
    for line in text.split('\n'):
        # Split on tab
        col = line.split('\t')

        # Remove invalid records
        if len(col) != 2:
            continue

        # Add to datatxt
        w = col[0].split("#")
        datatxt.append(w + [col[1].lower()])

    return datatxt


if __name__ == '__main__':
    # Configure details
    # warnings.filterwarnings("ignore")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.visible_device_list = "0"

    # Start tf session
    set_session(tf.Session(config=config))

    # Read file
    logging.info("Start reading caption file and pre-processing captions")
    text = read_file(captions_path)
    text = split_lines(text)

    # Create Dataframe with pandas
    df_txt = pd.DataFrame(text, columns=["image_idx", "caption_idx", "caption"])

    # Find unique image_idx
    uni_image_idx = np.unique(df_txt.image_idx.values)
    print("The number of unique file names : {}".format(len(uni_image_idx)))
    print("The distribution of the number of captions for each image:")
    print(Counter(Counter(df_txt.image_idx.values).values()))








from keras.preprocessing.image import load_img, img_to_array

npic = 5
npix = 224
target_size = (npix, npix, 3)

count = 1
fig = plt.figure(figsize=(10, 20))
for jpgfnm in uni_image_idx[:npic]:
    filename = dir_Flickr_jpg + '/' + jpgfnm
    captions = list(df_txt["caption"].loc[df_txt["filename"] == jpgfnm].values)
    image_load = load_img(filename, target_size=target_size)

    ax = fig.add_subplot(npic, 2, count, xticks=[], yticks=[])
    ax.imshow(image_load)
    count += 1

    ax = fig.add_subplot(npic, 2, count)
    plt.axis('off')
    ax.plot()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(captions))
    for i, caption in enumerate(captions):
        ax.text(0, i, caption, fontsize=20)
    count += 1
plt.show()


def set_seed(sd=123):
    from numpy.random import seed
    from tensorflow import set_random_seed
    import random as rn
    ## numpy random seed
    seed(sd)
    ## core python's random number
    rn.seed(sd)
    ## tensor flow's random number
    set_random_seed(sd)


def df_word(df_txt):
    vocabulary = []
    for txt in df_txt.caption.values:
        vocabulary.extend(txt.split())
    print('Vocabulary Size: %d' % len(set(vocabulary)))
    ct = Counter(vocabulary)
    dfword = pd.DataFrame({"word": ct.keys(), "count": ct.values()})
    dfword = dfword.sort("count", ascending=False)
    dfword = dfword.reset_index()[["word", "count"]]
    return (dfword)


dfword = df_word(df_txt)
dfword.head(3)

topn = 50


def plthist(dfsub, title="The top 50 most frequently appearing words"):
    plt.figure(figsize=(20, 3))
    plt.bar(dfsub.index, dfsub["count"])
    plt.yticks(fontsize=20)
    plt.xticks(dfsub.index, dfsub["word"], rotation=90, fontsize=20)
    plt.title(title, fontsize=20)
    plt.show()


plthist(dfword.iloc[:topn, :],
        title="The top 50 most frequently appearing words")
plthist(dfword.iloc[-topn:, :],
        title="The least 50 most frequently appearing words")

import string

text_original = "I ate 1000 apples and a banana. I have python v2.7. It's 2:30 pm. Could you buy me iphone7?"

print(text_original)
print("\nRemove punctuations..")


def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(None, string.punctuation)
    return (text_no_punctuation)


text_no_punctuation = remove_punctuation(text_original)
print(text_no_punctuation)

print("\nRemove a single character word..")


def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return (text_len_more_than1)


text_len_more_than1 = remove_single_character(text_no_punctuation)
print(text_len_more_than1)

print("\nRemove words with numeric values..")


def remove_numeric(text, printTF=False):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if printTF:
            print("    {:10} : {:}".format(word, isalpha))
        if isalpha:
            text_no_numeric += " " + word
    return (text_no_numeric)


text_no_numeric = remove_numeric(text_len_more_than1, printTF=True)
print(text_no_numeric)


def text_clean(text_original):
    text = remove_punctuation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)
    return (text)


for i, caption in enumerate(df_txt.caption.values):
    newcaption = text_clean(caption)
    df_txt["caption"].iloc[i] = newcaption

dfword = df_word(df_txt)
plthist(dfword.iloc[:topn, :],
        title="The top 50 most frequently appearing words")
plthist(dfword.iloc[-topn:, :],
        title="The least 50 most frequently appearing words")

from copy import copy


def add_start_end_seq_token(captions):
    caps = []
    for txt in captions:
        txt = 'startseq ' + txt + ' endseq'
        caps.append(txt)
    return (caps)


df_txt0 = copy(df_txt)
df_txt0["caption"] = add_start_end_seq_token(df_txt["caption"])
df_txt0.head(5)
del df_txt
