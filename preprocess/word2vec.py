import operator
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

START_TOKEN = 'START'
END_TOKEN = 'END'
PAD_TOKEN = 'PAD'
UNK_TOKEN = 'UNK'


class Word2Vector:
    def __init__(self, pretrained_path, dimensions):

        # Original vector size
        self.word_vec_dim = dimensions
        # Embedding size is two larger, see special tokens
        self.embedding_dim = dimensions + 2

        # Load the w2v model
        self.w2v = self.load_w2v(pretrained_path, self.word_vec_dim)
        self.ZERO_COLS = np.array([0.0, 0.0], dtype=float)

        # SPECIAL TOKENS:
        # For the start/end token, we add two new columns to the embedding and make one column one
        # This gives unique tokens that are not close to other vectors
        # Unkown token is the average vector of GLOVE
        # https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
        self.tokens = {START_TOKEN: np.concatenate((np.ones(self.word_vec_dim, dtype=float), np.array([1.0, 0.0]))),
                       END_TOKEN: np.concatenate((np.ones(self.word_vec_dim, dtype=float), np.array([0.0, 1.0]))),
                       PAD_TOKEN: np.ones(self.embedding_dim),
                       UNK_TOKEN: self.compute_average()}
        # We use np.pad for padding, with the pad value of 1 just like the PAD token
        self.pad_val = 1

        # Keep track of the known and unknown words for analysis
        self.knowns = defaultdict(int)
        self.unknowns = defaultdict(int)

    def load_w2v(self, path, dims):
        w2v = dict()
        with open(path, "r", encoding='UTF-8') as lines:
            for line in tqdm(lines, desc="Loading word2vec"):
                vec = line.split()

                # Check if what we read makes sense
                if len(vec[1:]) != dims:
                    raise ValueError("GloVe Word2Vec -- wrong dimensions! Expected %s., got %s., for word %s." % (
                        dims, len(vec[1:]), vec[0]))

                v = np.array(list(map(float, vec[1:])), dtype=float)
                w2v[vec[0]] = np.concatenate((v, self.ZERO_COLS))

        return w2v

    def compute_average(self):
        vecs = np.zeros((len(self.w2v), self.embedding_dim), dtype=np.float32)

        i = 0
        for vec in tqdm(self.w2v.values(), desc="Computing average word2vec"):
            vecs[i, :] = list(vec)
            i += 1

        return np.mean(vecs, axis=0)

    def get_vec(self, word):
        try:
            v = np.array(self.w2v[word])
            # Count as known if this not gave an exception
            self.knowns[word] += 1
        except KeyError:
            v = self.tokens['UNK']
            self.unknowns[word] += 1

        return v

    def embed_sentence(self, sentence, max_length):
        words = sentence.split()
        N = len(words) + 2  # Add two for the start and end token
        embedded = np.zeros((N, self.embedding_dim))
        # Add start and end token
        embedded[0, :] = self.tokens['START']
        embedded[-1, :] = self.tokens['END']

        for i, w in enumerate(words):
            v = self.get_vec(w)
            # i+1 as the first row was the start token
            if v.shape[0] == self.embedding_dim:
                embedded[i + 1, :] = v
            else:
                print(v)
                raise ValueError("Embedding is of the wrong size!")

        # Pad until max length
        embedded = np.pad(embedded, [(0, max_length - len(embedded)), (0, 0)], mode='constant',
                          constant_values=self.pad_val)

        return embedded

    def create_embeddings(self, captions, max_length):
        # Regrister tqdm for pandas
        tqdm.pandas(desc="Word2Vec embedding")

        # Embed each sentence
        embed_fn = lambda sent: self.embed_sentence(sent, max_length).flatten()
        embeddings = captions.progress_apply(embed_fn)

        return embeddings

    def analysis(self, dir_path):
        knowns = self.knowns
        unknowns = self.unknowns

        print("Total words: %d" % (len(knowns) + len(unknowns)))
        print("## Known words")
        print("There were %d unique known words, for a total of %d words" % (sum(knowns.values()), len(knowns)))
        print("## Unknowns words")
        print("There were %d unique unknowns words, for a total of %d words" % (sum(unknowns.values()), len(unknowns)))
        print("## Saving counts")

        # Save a dict of counts from largest to smallest
        def __save_sorted(fname, counts):
            with open(fname, "w+") as f:
                print("TOTAL\t%d" % (sum(counts.values())), file=f)
                print("UNIQUE\t%d" % (len(counts)), file=f)
                for k, c in sorted(counts.items(), key=operator.itemgetter(1), reverse=True):
                    print("%s\t%d" % (k, c), file=f)

        __save_sorted(os.path.join(dir_path, "knowns.tsv"), knowns)
        __save_sorted(os.path.join(dir_path, "unkowns.tsv"), unknowns)
