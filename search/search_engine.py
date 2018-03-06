import argparse

from collections import defaultdict
from math import log

import numpy as np
from annoy import AnnoyIndex

from search.dictionary import BilingualDictionary, MonolingualDictionary
from utils.document_frequencies import read_dfs
from utils.text_tools import normalize, tokenize


class SearchEngine:
    def __init__(self,
                 tf_function=lambda tf: (1 + log(tf, 10) if tf is not 0 else 0),
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
    def __init__(self, dictionary, df_file=None, df_options={}):
        super().__init__(df_file, df_options)

        self.dictionary = dictionary
        self.index = AnnoyIndex(dictionary.vector_dimensionality, metric='angular')
        self.documents = []

    def index_documents(self, documents):
        doc_tokens = []
        for i, document in enumerate(documents):
            self.documents.append(document)
            doc_tokens.append(tokenize(normalize(document)))

        self._init_word_weights_stopwords(doc_tokens, **self.word_weight_options)

        for i, tokens in enumerate(doc_tokens):
            self.index.add_item(i, self._vectorize(tokens=tokens))
        self.index.build(n_trees=10)

    def query_index(self, query, n_results=5):
        query_vector = self._vectorize(tokenize(normalize(query)), indexing=False)
        results, distances = self.index.get_nns_by_vector(query_vector,
                                                          n=n_results,
                                                          include_distances=True,
                                                          search_k=10*len(self.documents))
        return [(distance, self.documents[result]) for result, distance in zip(results, distances)]

    def _vectorize(self, tokens, indexing=True):
        vector = np.zeros((self.dictionary.vector_dimensionality,))
        for token in tokens:
            if token in self.stopwords:
                continue
            vector += self.dictionary.word_vector(token=token) * self.word_weights.get(token, self.default_word_weight)
        return vector


class BilingualEmbeddingSearchEngine(EmbeddingSearchEngine):
    def __init__(self, dictionary):
        super().__init__(dictionary=dictionary)

    def _vectorize(self, tokens, indexing=False):
        return np.sum(self.dictionary.word_vectors(tokens=tokens, reverse=not indexing), axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search Engine.')

    parser.add_argument('src_emb_file', type=str, help='File with document embeddings')
    parser.add_argument('tgt_emb_file', type=str, help='File with query embeddings', default=None)

    args = parser.parse_args()

    if args.tgt_emb_file is None:  # monolingual
        mono_dict = MonolingualDictionary(args.src_emb_file)
        search_engine = EmbeddingSearchEngine(dictionary=mono_dict)
    else:  # bilingual
        bi_dict = BilingualDictionary(args.src_emb_file, args.tgt_emb_file)
        search_engine = BilingualEmbeddingSearchEngine(dictionary=bi_dict)

    print('Type each sentence to index, followed by enter. When done, hit enter twice.')
    sentences = []
    sentence = input(">> ")
    while sentence:
        sentences.append(sentence)
        sentence = input(">> ")
    search_engine.index_documents(sentences)

    print('Type your query.')
    while True:
        query = input(">> ")
        print(search_engine.query_index(query))
