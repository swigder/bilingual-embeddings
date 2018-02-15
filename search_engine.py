import argparse

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
