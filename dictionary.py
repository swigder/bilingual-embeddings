import argparse

import os

import numpy as np
from gensim.models.keyedvectors import EuclideanKeyedVectors


class Dictionary:
    def word_vector(self, token):
        return self.word_vectors([token])[0]

    def word_vectors(self, tokens):
        pass

    def synonyms(self, src_word, topn=1):
        pass


class MonolingualDictionary(Dictionary):
    def __init__(self, emb_file, language=None):
        self.emb = EuclideanKeyedVectors.load_word2vec_format(emb_file, binary=False)
        self.vector_dimensionality = self.emb.vector_size
        self.language = language if language is not None else os.path.basename(emb_file).split('.')[0]

    def __contains__(self, token):
        return token in self.emb

    def safe_word_vector(self, token):
        if token not in self.emb:
            return np.zeros(shape=(self.vector_dimensionality,))
        return self.emb.word_vec(token)

    def word_vectors(self, tokens):
        return [self.safe_word_vector(token) for token in tokens]

    def synonyms(self, key, topn=1, vector=False):
        return self.emb.most_similar(key, topn=topn) if not vector else self.emb.similar_by_vector(key, topn=topn)


class BilingualDictionary(Dictionary):
    def __init__(self, src_emb_file, tgt_emb_file):
        assert os.path.exists(src_emb_file) and os.path.exists(tgt_emb_file)  # slow to open so don't want to waste time
        self.src_emb = MonolingualDictionary(emb_file=src_emb_file)
        self.tgt_emb = MonolingualDictionary(emb_file=tgt_emb_file)
        assert self.src_emb.vector_dimensionality == self.tgt_emb.vector_dimensionality
        self.vector_dimensionality = self.src_emb.vector_dimensionality

    def _embeddings(self, reverse):
        return (self.src_emb, self.tgt_emb) if not reverse else (self.tgt_emb, self.src_emb)

    def word_vectors(self, tokens, reverse=False):
        src_emb, _ = self._embeddings(reverse)
        return src_emb.word_vectors(tokens)

    def translate(self, src_word, topn=1, reverse=False):
        src_emb, tgt_emb = self._embeddings(reverse)
        src_vector = src_emb.word_vector(src_word)
        return tgt_emb.synonyms(src_vector, topn=topn, vector=True)

    def synonyms(self, src_word, topn=1, reverse=False):
        src_emb, _ = self._embeddings(reverse)
        return src_emb.synonyms(src_word, topn=topn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bilingual Dictionary.')

    parser.add_argument('src_emb_file', type=str, help='File with source embeddings')
    parser.add_argument('tgt_emb_file', type=str, help='File with target embeddings')
    parser.add_argument('-n', '--top_n', type=int, default=1, help='Number of translations to provide')
    # parser.add_argument('--mode', default='command', choices=['command', 'interactive'])
    # parser.add_argument()

    args = parser.parse_args()

    bi_dict = BilingualDictionary(args.src_emb_file, args.tgt_emb_file)

    reverse = False

    def language_direction():
        return ('{1} -> {0}' if reverse else '{0} -> {1}').format(bi_dict.src_emb.language, bi_dict.tgt_emb.language)

    print('Loaded dictionaries. Enter a word and hit enter to translate. Hit enter twice to reverse languages.')
    print(language_direction())
    while True:
        word = input(">> ")
        if not word:
            reverse = not reverse
            print('Reversed, new direction:', language_direction())
            continue
        print(bi_dict.translate(word, topn=args.top_n, reverse=reverse))
