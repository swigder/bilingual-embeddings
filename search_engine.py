from collections import defaultdict
from math import log

import numpy as np
from annoy import AnnoyIndex

from text_tools import normalize, tokenize
from utils import read_dfs


class SearchEngine:
    def __init__(self,
                 tf_function=lambda tf: (1 + log(tf, 10) if tf != 0 else 0),
                 df_options={}):
        self.tf_function = tf_function
        self.word_weight_options = df_options
        self.word_weights = defaultdict(int)
        self.stopwords = set()
        if 'df_file' in df_options and df_options['df_file'] is not None:
            self._init_word_weights_stopwords(**df_options)

    def index_documents(self, documents):
        pass

    def query_index(self, query, n_results=5):
        pass

    def _init_word_weights_stopwords(self, documents=None, df_file=None,
                                     df_cutoff=.8,
                                     df_to_weight=lambda df, num_docs: log(num_docs / df, 10),
                                     default_df_fn=lambda dfs: np.average(list(dfs))):
        if len(self.word_weights) > 0:
            return

        assert (documents is None) != (df_file is None)  # documents xor df_file
        dfs = defaultdict(int)
        if documents is not None:
            num_docs = len(documents)
            for document_tokens in documents:
                for token in set(document_tokens):
                    dfs[token] += 1
        else:
            dfs, num_docs = read_dfs(df_file)

        df_cutoff = int(df_cutoff * num_docs)
        self.stopwords = set([token for token, df in dfs.items() if df >= df_cutoff])
        self.word_weights = {token: df_to_weight(df, num_docs) for token, df in dfs.items()}
        self.default_word_weight = default_df_fn(list(self.word_weights.values()))


class EmbeddingSearchEngine(SearchEngine):
    def __init__(self, dictionary, tf_function=None, df_options={}):
        super().__init__(tf_function=tf_function, df_options=df_options)

        self.dictionary = dictionary
        self.index = AnnoyIndex(dictionary.vector_dimensionality, metric='angular')
        self.documents = []
        self.weighted_word_vector_cache = {}

    def index_documents(self, documents):
        doc_tokens = []
        for i, document in enumerate(documents):
            self.documents.append(document)
            doc_tokens.append(tokenize(normalize(document)))

        self._init_word_weights_stopwords(doc_tokens, **self.word_weight_options)

        for i, tokens in enumerate(doc_tokens):
            self.index.add_item(i, self._vectorize(tokens=tokens, indexing=True))
        self.index.build(n_trees=10)

    def query_index(self, query, n_results=5):
        query_vector = self._vectorize(tokenize(normalize(query)), indexing=False)
        results, distances = self.index.get_nns_by_vector(query_vector,
                                                          n=n_results,
                                                          include_distances=True,
                                                          search_k=10 * len(self.documents))
        return [(distance, self.documents[result]) for result, distance in zip(results, distances)]

    def _vectorize(self, tokens, indexing):
        vector = np.zeros((self.dictionary.vector_dimensionality,))
        for token in tokens:
            if token in self.stopwords:
                continue
            vector += self._weighted_word_vector(token)
        return vector

    def _weighted_word_vector(self, word):
        if word in self.weighted_word_vector_cache:
            return self.weighted_word_vector_cache[word]
        weighted_word = self.dictionary.word_vector(token=word) * self.word_weights.get(word, self.default_word_weight)
        self.weighted_word_vector_cache[word] = weighted_word
        return weighted_word


class BilingualEmbeddingSearchEngine(EmbeddingSearchEngine):
    def __init__(self, dictionary, doc_lang, query_lang):
        super().__init__(dictionary=dictionary)
        self.doc_lang = doc_lang
        self.query_lang = query_lang

    def _vectorize(self, tokens, indexing):
        if indexing:  # document language, use df
            # return np.sum(self.dictionary.word_vectors(tokens=tokens, lang=self.doc_lang), axis=0)
            return EmbeddingSearchEngine._vectorize(self, tokens, indexing)
        else:  # query language, df not available
            vector = np.sum(self.dictionary.word_vectors(tokens=tokens, lang=self.query_lang), axis=0)
            oov_tokens = [token for token in tokens if token not in self.dictionary.dictionaries[self.query_lang]]
            print('OOV', oov_tokens)
            vector += np.sum(self.dictionary.word_vectors(tokens=oov_tokens, lang=self.doc_lang), axis=0)
            return vector
