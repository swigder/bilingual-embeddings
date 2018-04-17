import os

import numpy as np
from gensim.models.deprecated.keyedvectors import EuclideanKeyedVectors
import fastText


class Dictionary:
    def word_vector(self, token):
        return self.word_vectors([token])[0]

    def word_vectors(self, tokens):
        pass

    def synonyms(self, src_word, topn=1):
        pass


class SubwordDictionary(Dictionary):
    def __init__(self, emb_file, language=None):
        self.emb = fastText.load_model(path=emb_file)
        self.vector_dimensionality = len(self.emb.get_word_vector(''))
        self.language = language if language is not None else os.path.basename(emb_file).split('.')[0]

    def __contains__(self, token):
        return self.emb.get_word_id(token) >= 0

    def safe_word_vector(self, token):
        return self.emb.get_word_vector(token)

    def word_vectors(self, tokens):
        return [self.safe_word_vector(token) for token in tokens]

    def synonyms(self, key, topn=1, vector=False):
        raise NotImplementedError()


class MonolingualDictionary(Dictionary):
    def __init__(self, emb_file, language=None):
        try:
            self.emb_file = emb_file
            self.emb = EuclideanKeyedVectors.load_word2vec_format(emb_file,
                                                                  binary=os.path.splitext(emb_file)[1] == '.bin')
        except Exception:
            print('Error with embed file:', emb_file)
            raise
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
    def __init__(self, src_emb_file, tgt_emb_file, src_lang='src', tgt_lang='tgt', default_lang=None, subword=False):
        assert os.path.exists(src_emb_file) and os.path.exists(tgt_emb_file)  # slow to open so don't want to waste time
        cls = MonolingualDictionary if not subword else SubwordDictionary
        self.dictionaries = {src_lang: cls(emb_file=src_emb_file, language=src_lang),
                             tgt_lang: cls(emb_file=tgt_emb_file, language=tgt_lang)}
        self.default_lang = default_lang
        assert self.dictionaries[src_lang].vector_dimensionality == self.dictionaries[tgt_lang].vector_dimensionality
        self.vector_dimensionality = self.dictionaries[src_lang].vector_dimensionality

    def word_vectors(self, tokens, lang=None):
        lang = lang or self.default_lang
        return self.dictionaries[lang].word_vectors(tokens)

    def translate(self, src_word, src_lang, topn=1):
        src_lang = src_lang or self.default_lang
        src_emb = self.dictionaries[src_lang]
        tgt_emb = self.dictionaries[next(iter(set(self.dictionaries.keys()).difference({src_lang})))]
        src_vector = src_emb.word_vector(src_word)
        return tgt_emb.synonyms(src_vector, topn=topn, vector=True)

    def synonyms(self, src_word, topn=1, lang=None):
        lang = lang or self.default_lang
        return self.dictionaries[lang].synonyms(src_word, topn=topn)
