import string
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


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


def make_df_word(df_txt, analysis=True):
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


def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(str.maketrans('', '', string.punctuation))
    return text_no_punctuation


def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word

    # Remove the first space again
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


def text_clean(text_original):
    text = remove_punctuation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)
    # Remove the starting space that was added
    # text = text.lstrip()
    return text


def read_captions(captions_path):
    text = read_file(captions_path)
    text = split_lines(text)

    # Create Dataframe with pandas
    df_txt = pd.DataFrame(text, columns=["image_idx", "caption_idx", "caption"])

    return df_txt


def clean_captions(df_txt):
    for i, caption in enumerate(tqdm(df_txt.caption.values, desc='Cleaning captions')):
        newcaption = text_clean(caption)
        df_txt["caption"].iloc[i] = newcaption

    return df_txt


def train_sp(in_path, out_path, vocab_size):
    # model_prefix = "trained_sp"
    spm.SentencePieceTrainer.Train(f'--input={in_path} '
                                   f'--model_prefix={out_path} '
                                   f'--vocab_size={vocab_size} '
                                   f'--bos_id=0 '
                                   f'--unk_id=1 '
                                   f'--pad_id=2 '
                                   f'--eos_id=3 ')
    return


def load_sp(in_path):
    # Load trained model
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{in_path}')

    return sp


def add_padding(df_txt, maxlength, space_tok=2):
    # Pad each caption to maxlength
    df_res = df_txt.copy()
    for i, caption in enumerate(tqdm(df_txt.caption.values, desc='Post padding to equal length')):
        df_res['caption'].iloc[i] = pad_sequences([caption], maxlen=maxlength, value=space_tok,
                                                  padding='post').flatten()

    return df_res


def encode_caption_as_ids(df_txt, sp):
    df_res = df_txt.copy()
    sp.SetEncodeExtraOptions(extra_option='bos:eos')
    # Encode with sp to idx in the vocab
    for i, caption in enumerate(tqdm(df_txt.caption.values, desc='Encoding text captions as SP indices')):
        df_res['caption'].iloc[i] = sp.EncodeAsIds(caption)

    return df_res


def main2():
    """ Steps to process text -- is implemented in the prepo.py file"""
    vocab = 2048
    df_txt = read_captions("")
    df_txt = clean_captions(df_txt)

    # Save all of the captions as .txt for SP
    np.savetxt(r'text_dataframe.txt', df_txt["caption"], fmt='%s')

    # Train the SP and save it
    train_sp('text_dataframe.txt', 'trained_sp', vocab)
    sp = load_sp('trained_sp')

    # Whoops this pickle is to big
    # result = onehot_encode(df_txt, sp, vocab)
    # result.to_pickle('encoded_captions.p')
