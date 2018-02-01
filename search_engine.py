import argparse

import numpy as np
from annoy import AnnoyIndex
from nltk import word_tokenize

from dictionary import BilingualDictionary, MonolingualDictionary


class SearchEngine:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.index = AnnoyIndex(dictionary.vector_dimensionality)
        self.documents = []

    def index_documents(self, documents):
        i = self.index.get_n_items()
        for document in documents:
            self.index.add_item(i, self._vectorize(tokens=word_tokenize(document)))
            self.documents.append(document)
            i += 1
        self.index.build(n_trees=5)

    def query_index(self, query):
        query_vector = self._vectorize(word_tokenize(query))
        result = self.index.get_nns_by_vector(query_vector, n=1)[0]
        return self.documents[result]

    def _vectorize(self, tokens):
        return np.sum(self.dictionary.word_vectors(tokens=tokens), axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search Engine.')

    parser.add_argument('emb_file', type=str, help='File with embeddings')

    args = parser.parse_args()

    mono_dict = MonolingualDictionary(args.emb_file)
    search_engine = SearchEngine(dictionary=mono_dict)

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
