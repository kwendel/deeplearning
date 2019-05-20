# Inspired of https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import string
import sentencepiece as spm

from keras.preprocessing.sequence import pad_sequences
from keras.backend.tensorflow_backend import set_session
from collections import Counter

# Path to captions file
captions_path = "C:/Users/User/Downloads/Flickr8k_text/Flickr8k.lemma.token.txt"

# For analysis
analysis = True
topn = 50


def read_file(path):
    """"
    Reads the file specified in path
    """
    file = open(path, 'r')
    text = file.read()
    file.close()
    return text


def split_lines(text):
    """"
    Splits each line in the specified text into different cols.
    """
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


def set_seed(sd=42):
    from numpy.random import seed
    from tensorflow import set_random_seed
    import random as rn
    # numpy random seed
    seed(sd)
    # core python's random number
    rn.seed(sd)
    # tensor flow's random number
    set_random_seed(sd)


def make_df_word(df_txt):
    """"
    Creates a data frame for the words
    """
    vocabulary = []
    for txt in df_txt.caption.values:
        vocabulary.extend(txt.split())

    if analysis:
        print('Vocabulary Size: %d' % len(set(vocabulary)))

    ct = Counter(vocabulary)

    dfword = pd.DataFrame({"word": list(ct.keys()), "count": list(ct.values())})
    dfword = dfword.sort_values("count", ascending=False)
    dfword = dfword.reset_index()[["word", "count"]]
    return dfword


def plot_hist(dfsub, title="Give a title next time"):
    plt.figure(figsize=(20, 3))
    plt.bar(dfsub.index, dfsub["count"])
    plt.yticks(fontsize=20)
    plt.xticks(dfsub.index, dfsub["word"], rotation=90, fontsize=20)
    plt.title(title, fontsize=20)
    plt.show()


def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(str.maketrans('', '', string.punctuation))
    return text_no_punctuation


def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return text_len_more_than1


def remove_numeric(text, printTF=False):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if printTF:
            print("    {:10} : {:}".format(word, isalpha))
        if isalpha:
            text_no_numeric += " " + word
    return text_no_numeric


def add_start_end_seq_token(text):
    return 'startseq ' + text + ' endseq'


def text_clean(text_original):
    text = remove_punctuation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)
    # text = add_start_end_seq_token(text)
    return text


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


if __name__ == '__main__':
    # Configure details
    # warnings.filterwarnings("ignore")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.visible_device_list = "0"

    # Start tf session
    set_session(tf.Session(config=config))

    # Read file
    print("Start reading file\n")
    text = read_file(captions_path)
    text = split_lines(text)

    # Create Dataframe with pandas
    df_txt = pd.DataFrame(text, columns=["image_idx", "caption_idx", "caption"])
    df_word = make_df_word(df_txt)

    if analysis:
        print("Start file analysis\n")

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

    # Clean text
    print("Start cleaning captions")
    l = len(df_txt.caption)
    print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for i, caption in enumerate(df_txt.caption.values):
        newcaption = text_clean(caption)
        df_txt["caption"].iloc[i] = newcaption
        print_progress_bar(i + 1, l, prefix='Progress cleaning captions:', suffix='Complete', length=50, decimals=1)

    # Analyze cleaned data
    if analysis:
        dfword = make_df_word(df_txt)
        plot_hist(dfword.iloc[:topn, :], title="The top " + str(topn) + " most frequently appearing words")
        plot_hist(dfword.iloc[-topn:, :], title="The least " + str(topn) + " most frequently appearing words")

    # Save to txt for encoding
    np.savetxt(r'text_dataframe.txt', df_txt["caption"], fmt='%s')

    # Train SentencePiece Model
    model_prefix = "trained_sp"
    spm.SentencePieceTrainer.Train(f'--input=text_dataframe.txt --model_prefix={model_prefix} --vocab_size=2048 --pad_id=3 --extra_options=bos:eos')

    # Load trained model
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{model_prefix}.model')

    # Encode all captions with sentencepiece
    print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
    maxlen = 0
    for i, caption in enumerate(df_txt.caption.values):
        newcaption = sp.encode_as_ids(caption)
        df_txt["caption"].iloc[i] = newcaption
        if maxlen < len(newcaption):
            maxlen = len(newcaption)
        print_progress_bar(i + 1, l, prefix='Progress enconding captions:', suffix='Complete', length=50, decimals=1)

    for i, caption in enumerate(df_txt.caption.values):
        df_txt["caption"].iloc[i] = pad_sequences([caption], maxlen=maxlen, value=3)
        
    # Write to pickle file
    df_txt.to_pickle(path="encoded_captionsencoded_captions.p")
