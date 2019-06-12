import numpy as np
from gensim.models import KeyedVectors

from .word2vec import PAD_TOKEN


class Vec2Word:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load_model(cls, path, dimension):
        return cls(KeyedVectors(vector_size=dimension).load(path))

    @classmethod
    def create_with(cls, model):
        return cls(model)

    def get_vec(self, word):
        return self.model.wv[word]

    def vec2word(self, vec, topn):
        # Returns list with topn tuples of (word,similarity_score)
        vectors = self.model.wv.similar_by_vector(vec, topn=topn)

        # Flatten the list if we only asked for one
        if topn == 1:
            return vectors[0]

        return vectors

    def matrix2sent(self, matrix):
        # Get the words
        getter = lambda x: self.vec2word(x, topn=1)
        decoded = np.apply_along_axis(getter, 1, matrix)

        # Join all the non pad tokens
        sent = " ".join([w if w != PAD_TOKEN else "" for (w, s) in decoded])

        # Also join decoded as (str, score) as this is easier for handling later on
        sent_with = " ".join(["({},{.2f})".format(w, s) for (w, s) in decoded])

        return sent_with, sent

    @staticmethod
    def create_and_save(w2v, special_tokens, dimension, path):
        entities = list(w2v.keys())
        vectors = list(w2v.values())

        # Add our special tokens to the sets
        for name, vec in special_tokens.items():
            entities.append(name)
            vectors.append(vec)

        model = KeyedVectors(vector_size=dimension)
        model.add(entities, vectors)

        with open(path, "wb+") as f:
            model.save(f)

        return Vec2Word.create_with(model)
