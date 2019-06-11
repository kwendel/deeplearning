import numpy as np
from tqdm import tqdm

w2v_path = "models/pretrained/glove.6B/glove.6B.50d.txt"
captions_path = 'dataset/Flickr8k/prepro/cleaned_captions.pkl'

# Unkown token is the average vector of GLOVE
# https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
UNK_VEC = np.array([-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965,
                    -0.08389011, 0.33359843, 0.16045167, 0.03867431, 0.17833012,
                    0.04696583, -0.00285802, 0.29099807, 0.04613704, -0.20923874,
                    -0.06613114, -0.06822549, 0.07665912, 0.3134014, 0.17848536,
                    -0.1225775, -0.09916984, -0.07495987, 0.06413227, 0.14441176,
                    0.60894334, 0.17463093, 0.05335403, -0.01273871, 0.03474107,
                    -0.8123879, -0.04688699, 0.20193407, 0.2031118, -0.03935686,
                    0.06967544, -0.01553638, -0.03405238, -0.06528071, 0.12250231,
                    0.13991883, -0.17446303, -0.08011883, 0.0849521, -0.01041659,
                    -0.13705009, 0.20127155, 0.10069408, 0.00653003, 0.01685157,
                    ], dtype=float)

# For the start/end token, we add two new columns to the embedding and make these tokens 1 in one column
# This gives unique tokens that are not close to other vectors
ZERO_COLS = np.array([0.0, 0.0], dtype=float)
START_VEC = np.concatenate((np.ones(50, dtype=float), np.array([1.0, 0.0])))
END_VEC = np.concatenate((np.ones(50, dtype=float), np.array([0.0, 1.0])))
UNK_VEC = np.concatenate((UNK_VEC, ZERO_COLS))

# This makes our embedding dimension 50 (glove) + 2 (extra)
VEC_DIM = 52


# Also we add the zero_cols to the word2vec embedding


# We use np.pad for padding
# PAD_VEC = np.zeros((1, 50 + 2))


def compute_average(w2v, vec_dim=VEC_DIM):
    vecs = np.zeros((len(w2v), vec_dim), dtype=np.float32)

    i = 0
    for vec in w2v.values():
        vecs[i, :] = list(vec)
        i += 1

    return np.mean(vecs, axis=0)


def load_w2v(path):
    # For now, hardcode the GloVe 50 dimension
    dims = 50

    w2v = dict()
    with open(path, "r", encoding='UTF-8') as lines:
        for line in lines:
            vec = line.split()

            # Check if what we read makes sense
            if len(vec[1:]) != dims:
                raise ValueError("GloVe Word2Vec -- wrong dimensions! Expected %s., got %s., for word %s." % (dims, len(vec[1:]), vec[0]))

            v = np.array(list(map(float, vec[1:])), dtype=float)
            w2v[vec[0]] = np.concatenate((v, ZERO_COLS))
    return w2v


def embed_sentence(w2v, sentence, max_length, vec_dim=VEC_DIM):
    words = sentence.split()
    N = len(words) + 2  # Add two for the start and end token
    embedded = np.zeros((N, vec_dim))
    # Add start and end token
    embedded[0, :] = START_VEC
    embedded[-1, :] = END_VEC

    def __getvec(w):
        try:
            v = np.array(w2v[w])
        except KeyError:
            v = UNK_VEC

        return v

    for i, w in enumerate(words):
        v = __getvec(w)
        # i+1 as the first row was the start token
        if v.shape[0] == 52:
            embedded[i + 1, :] = v
        else:
            print(v)
            break

    # Pad until max length
    embedded = np.pad(embedded, [(0, max_length - len(embedded)), (0, 0)], mode='constant', constant_values=1.0)

    return embedded


def create_embeddings(df_captions, max_length):
    # Add two to the max length for start and end token, this will give the longest sentence no padding
    max_length = max_length + 2

    # Load word2vec dictionary
    w2v = load_w2v(w2v_path)

    # Regrister tqdm for pandas
    tqdm.pandas(desc="Word2Vec embedding")

    # Embed each sentence
    embed_fn = lambda row: embed_sentence(w2v, row['caption'], max_length).flatten()
    embeddings = df_captions.progress_apply(embed_fn, axis=1)

    # Save as a new column and return the df
    df_captions['word2vec'] = embeddings
    return df_captions


# if __name__ == '__main__':
# captions = pickle.load(open(captions_path, 'rb'))
# w2v = load_w2v(w2v_path)
# # TODO: set this to the maximum amount of words, +2 for start and end token
# max_length = 32
#
# # Regrister tqdm for pandas
# tqdm.pandas(desc="Word2Vec embedding")
#
# # Embed each sentence
# embed_fn = lambda row: embed_sentence(w2v, row['caption'], max_length).flatten()
# embeddings = captions.progress_apply(embed_fn, axis=1)
# # embeddings = df_captions.apply(embed_fn)
#
# # Save as a new column and return the df
# captions['word2vec'] = embeddings
