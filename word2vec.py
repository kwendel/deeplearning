import pickle

import numpy as np
from tqdm import tqdm

w2v_path = "models/pretrained/glove.6B/glove.6B.50d.txt"
captions_path = 'dataset/Flickr8k/prepro/cleaned_captions.pkl'

# Unkown token is the average vector of GLOVE
# https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
UNK_VEC = np.array([-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011,
                    0.33359843, 0.16045167, 0.03867431, 0.17833012, 0.04696583, -0.00285802,
                    0.29099807, 0.04613704, -0.20923874, -0.06613114, -0.06822549, 0.07665912,
                    0.3134014, 0.17848536, -0.1225775, -0.09916984, -0.07495987, 0.06413227,
                    0.14441176, 0.60894334, 0.17463093, 0.05335403, -0.01273871, 0.03474107,
                    -0.8123879, -0.04688699, 0.20193407, 0.2031118, -0.03935686, 0.06967544,
                    -0.01553638 - 0.03405238, -0.06528071, 0.12250231, 0.13991883, -0.17446303,
                    -0.08011883, 0.0849521, -0.01041659, -0.13705009, 0.20127155, 0.10069408,
                    0.00653003, 0.01685157])

START_VEC = np.ones((1, 50))
END_VEC = np.ones((1, 50)) + 1
PAD_VEC = np.zeros((1, 50))


def compute_average(w2v: dict, vec_dim=50):
    vecs = np.zeros((len(w2v), vec_dim), dtype=np.float32)

    i = 0
    for vec in w2v.values():
        vecs[i, :] = list(vec)
        i += 1

    return np.mean(vecs, axis=0)


def load_w2v(path):
    w2v = dict()
    with open(path, "rb") as lines:
        for line in lines:
            vec = line.decode('UTF-8').split()
            w2v[vec[0]] = list(map(float, vec[1:]))

    return w2v


def embed_sentence(w2v, sentence, vec_dim=50, max_length=20):
    words = sentence.split()
    N = len(words) + 2  # Add two for the start and end token
    embedded = np.zeros((len(words) + 2, vec_dim))
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
        # The first word contains a leading whitespace
        # We check in this way as this is faster
        if i == 0:
            w = w.lstrip()

        v = __getvec(w)
        # i+1 as the first row was the start token
        embedded[i+1, :] = v

    # Pad until max length
    embedded = np.pad(embedded, [(0, max_length - len(embedded)), (0, 0)], mode='constant', constant_values=0)

    return embedded


if __name__ == '__main__':
    captions = pickle.load(open(captions_path, 'rb'))
    w2v = load_w2v(w2v_path)
    # TODO: set this to the maximum amount of words, +2 for start and end token
    max_length = 20 + 2

    for cap in tqdm(captions['caption'], desc="Word2Vec embedding"):
        embedded = embed_sentence(w2v, cap)
        breakpoint()
