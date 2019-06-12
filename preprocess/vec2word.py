import numpy as np
from gensim.models import KeyedVectors


class Vec2Word:
    def __init__(self, path, dimension=52):
        self.model = KeyedVectors(vector_size=dimension).load(path)

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
        sent = " ".join([w if w != "PAD" else "" for (w, s) in decoded])

        return decoded, sent

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
