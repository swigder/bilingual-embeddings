import argparse

from collections import defaultdict
from math import sqrt

import numpy as np
from annoy import AnnoyIndex
from nltk import word_tokenize
from nltk.corpus import stopwords

from dictionary import BilingualDictionary, MonolingualDictionary


class SearchEngine:
    def index_documents(self, documents):
        pass

    def query_index(self, query, n_results=5):
        pass


class EmbeddingSearchEngine(SearchEngine):
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.index = AnnoyIndex(dictionary.vector_dimensionality, metric='angular')
        self.documents = []

    def index_documents(self, documents):
        i = self.index.get_n_items()
        to_remove = stopwords.words('english')
        for document in documents:
            tokens = [word for word in word_tokenize(document) if word not in to_remove]
            self.index.add_item(i, self._vectorize(tokens=tokens))
            self.documents.append(document)
            i += 1
        self.index.build(n_trees=5)

    def query_index(self, query, n_results=5):
        query_vector = self._vectorize(word_tokenize(query), indexing=False)
        results, distances = self.index.get_nns_by_vector(query_vector, n=n_results, include_distances=True)
        return [(distance, self.documents[result]) for result, distance in zip(results, distances)]

    def _vectorize(self, tokens, indexing=True):
        return np.sum(self.dictionary.word_vectors(tokens=tokens), axis=0)


class BilingualEmbeddingSearchEngine(EmbeddingSearchEngine):
    def __init__(self, dictionary):
        super().__init__(dictionary=dictionary)

    def _vectorize(self, tokens, indexing=False):
        return np.sum(self.dictionary.word_vectors(tokens=tokens, reverse=not indexing), axis=0)


class TfIdfSearchEngine(SearchEngine):
    def __init__(self):
        self.index = {}
        self.documents = []

    def index_documents(self, documents):
        to_remove = stopwords.words('english')
        to_remove.append('.')
        for i, document in enumerate(documents):
            tokens = [word for word in word_tokenize(document) if word not in to_remove]
            self.documents.append((document, tokens, self._norm(tokens)))
            for token in tokens:
                if token not in self.index:
                    self.index[token] = [i]
                elif self.index[token][-1] != i:
                    self.index[token].append(i)

    def query_index(self, query, n_results=5):
        query_tokens = [word for word in word_tokenize(query) if word in self.index]
        dimensions = {token: i for (i, token) in enumerate(list(set(query_tokens)))}
        query_vector = self._vectorize(query_tokens, dimensions)
        query_norm = self._norm(query_tokens)
        processed = set()
        top_hits = [(0, None)] * n_results  # using simple array with assumption that n_results is small
        dfs = {token: len(self.index[token]) for token in dimensions.keys()}
        for token in dimensions.keys():
            for document_id in self.index[token]:
                if document_id in processed:
                    continue
                processed.add(document_id)
                document, document_tokens, document_norm = self.documents[document_id]
                document_vector = self._vectorize(document_tokens, dimensions)
                similarity = sum([dfs[dim] * query_vector[i] * document_vector[i] for dim, i in dimensions.items()])
                similarity /= (query_norm * document_norm)
                if similarity > top_hits[0][0]:
                    del top_hits[0]
                    insert_location = 0
                    for score, _ in top_hits:
                        if similarity < score:
                            break
                        insert_location += 1
                    top_hits.insert(insert_location, (similarity, document_id))
        return [(score, self.documents[doc_id][0]) for (score, doc_id) in reversed(top_hits)]

    @staticmethod
    def _vectorize(tokens, dimensions):
        vector = [0] * len(dimensions)
        for token in tokens:
            if token in dimensions:
                vector[dimensions[token]] += 1
        return vector

    @staticmethod
    def _norm(tokens):
        dimensions = defaultdict(int)
        for token in tokens:
            dimensions[token] += 1
        return sqrt(sum([d**2 for d in dimensions.values()]))


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
