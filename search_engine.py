import argparse

from collections import defaultdict
from math import sqrt, log

import numpy as np
from annoy import AnnoyIndex
from nltk import word_tokenize

from dictionary import BilingualDictionary, MonolingualDictionary
from document_frequencies import read_dfs


class SearchEngine:
    def __init__(self, df_file=None, df_options={}):
        self.df_options = df_options
        self.df = defaultdict(int)
        if df_file is None:
            self.stopwords = set()
        else:
            df, num_docs = read_dfs(df_file)
            self._init_df_stopwords(df=df, num_docs=num_docs, **self.df_options)

    def index_documents(self, documents):
        pass

    def query_index(self, query, n_results=5):
        pass

    def _init_df_stopwords(self, documents=None, df=None, num_docs=None,
                           df_cutoff=.8,
                           smoothing_fn=lambda num_docs: int(sqrt(num_docs)),
                           default_df_fn=lambda dfs, smoothing: np.average(list(dfs)),
                           percentiles=None):
        if len(self.df) > 0:
            return

        assert (documents is None) != (df is None)  # documents xor df
        if documents is not None:
            num_docs = len(documents)
            for document_tokens in documents:
                for token in set(document_tokens):
                    self.df[token] += 1
        else:
            self.df = df

        if percentiles:
            for token, df in self.df.items():
                self.df[token] = int(df / num_docs * percentiles) + 1

        smoothing = smoothing_fn(num_docs)
        for token, df in self.df.items():
            self.df[token] = df + smoothing

        df_cutoff = int(df_cutoff * (num_docs + smoothing))
        self.stopwords = set([token for token, df in self.df.items() if df >= df_cutoff])

        self.default_df = default_df_fn(list(self.df.values()), smoothing)


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
            doc_tokens.append(word_tokenize(document))

        self._init_df_stopwords(doc_tokens, **self.df_options)

        for i, tokens in enumerate(doc_tokens):
            self.index.add_item(i, self._vectorize(tokens=tokens))
        self.index.build(n_trees=10)

    def query_index(self, query, n_results=5):
        query_vector = self._vectorize(word_tokenize(query), indexing=False)
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
            df = self.df.get(token, self.default_df)
            if df != 0:
                vector += self.dictionary.word_vector(token=token) / df
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
