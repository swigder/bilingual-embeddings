from collections import defaultdict
from math import sqrt

from nltk import word_tokenize
from nltk.corpus import stopwords

from search_engine import SearchEngine


class TfIdfSearchEngine(SearchEngine):
    def __init__(self):
        self.index = {}
        self.documents = []

    def index_documents(self, documents):
        to_remove = stopwords.words('english')
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
        query_norm = self._norm(query_tokens)
        processed = set()
        top_hits = [(0, None)] * n_results  # using simple array with assumption that n_results is small
        for token in set(query_tokens):
            for document_id in self.index[token]:
                if document_id in processed:
                    continue
                processed.add(document_id)
                document, document_tokens, document_norm = self.documents[document_id]
                dimensions = self._dimensions(query_tokens + document_tokens)
                dfs = {token: len(self.index[token]) for token in dimensions.keys()}
                query_vector = self._vectorize(query_tokens, dimensions)
                document_vector = self._vectorize(document_tokens, dimensions)
                similarity = sum([query_vector[i] * document_vector[i] / dfs[dim] for dim, i in dimensions.items()])
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
    def _dimensions(tokens):
            return {token: i for (i, token) in enumerate(list(set(tokens)))}

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