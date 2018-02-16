import argparse

from collections import defaultdict
from math import sqrt

import numpy as np
from annoy import AnnoyIndex
from nltk import word_tokenize

from dictionary import BilingualDictionary, MonolingualDictionary


class SearchEngine:
    def __init__(self):
        self.df = defaultdict(int)
        self.stopwords = set()

    def index_documents(self, documents):
        pass

    def query_index(self, query, n_results=5):
        pass

    def _init_df_stopwords(self, documents):
        smoothing = sqrt(len(documents))

        for document_tokens in documents:
            for token in set(document_tokens):
                self.df[token] += 1
        for token, df in self.df.items():
            self.df[token] = df + smoothing

        max_df = .9 * (len(documents) + smoothing)
        self.stopwords = set([token for token, df in self.df.items() if df >= max_df])


class EmbeddingSearchEngine(SearchEngine):
    def __init__(self, dictionary):
        super().__init__()

        self.dictionary = dictionary
        self.index = AnnoyIndex(dictionary.vector_dimensionality, metric='angular')
        self.documents = []
        self.default_df = 1

    def index_documents(self, documents):
        doc_tokens = []
        for i, document in enumerate(documents):
            self.documents.append(document)
            doc_tokens.append(word_tokenize(document))

        self._init_df_stopwords(doc_tokens)
        self.default_df = np.average(list(self.df.values()))

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
            vector += self.dictionary.word_vector(token=token) / self.df.get(token, self.default_df)
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
