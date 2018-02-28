from collections import defaultdict
from math import sqrt

from search_engine import SearchEngine
from text_tools import tokenize, normalize


class CosineSimilaritySearchEngine(SearchEngine):
    def __init__(self, tf_idf_options={}):
        super().__init__(**tf_idf_options)
        self.index = {}
        self.documents = []
        self.doc_tokens = []
        self.doc_vectors = []
        self.doc_norms = []

    def index_documents(self, documents):
        self.documents = list(documents)
        doc_tokens = []
        for document in documents:
            doc_tokens.append(tokenize(normalize(document)))
        self._init_word_weights_stopwords(doc_tokens, **self.word_weight_options)
        self.doc_tokens = [[token for token in document if token not in self.stopwords] for document in doc_tokens]
        self.doc_vectors = [self._vectorize(tokens, weighted=True) for tokens in self.doc_tokens]
        self.doc_norms = [self._norm(tokens) for tokens in doc_tokens]

        for i, document in enumerate(self.doc_tokens):
            for token in document:
                if token not in self.index:
                    self.index[token] = [i]
                elif self.index[token][-1] != i:
                    self.index[token].append(i)

    def query_index(self, query, n_results=5):
        query_tokens = [word for word in tokenize(normalize(query)) if word in self.index]
        query_norm = self._norm(query_tokens)
        query_vector = self._vectorize(query_tokens)
        query_weights = {token: count * self.word_weights[token] for token, count in query_vector.items()}
        processed = set()
        top_hits = [(0, None)] * n_results  # using simple array with assumption that n_results is small
        for token in set(query_tokens):
            for document_id in self.index[token]:
                if document_id in processed:
                    continue
                processed.add(document_id)
                document_tokens, document_norm = self.doc_tokens[document_id], self.doc_norms[document_id]
                document_vector = self.doc_vectors[document_id]
                similarity = self._similarity_unnorm(unweighted_vector=query_vector,
                                                     weighted_vector=document_vector,
                                                     weights=query_weights)
                similarity /= (query_norm * document_norm)
                if similarity > top_hits[0][0]:
                    del top_hits[0]
                    insert_location = 0
                    for score, _ in top_hits:
                        if similarity < score:
                            break
                        insert_location += 1
                    top_hits.insert(insert_location, (similarity, document_id))
        return [(score, self.documents[doc_id]) for (score, doc_id) in reversed(top_hits) if doc_id is not None]

    def _dimensions(self, tokens):
        cleaned_tokens = list(set([token for token in tokens if token in self.word_weights]))
        return {token: i for (i, token) in enumerate(cleaned_tokens)}

    def _vectorize(self, tokens, weighted=False):
        vector = defaultdict(int)
        for token in tokens:
            vector[token] += 1
        if weighted:
            for token in vector.keys():
                vector[token] *= self.word_weights[token]
        return dict(vector)

    def _similarity_unnorm(self, unweighted_vector, weighted_vector, weights):
        vector_sum = 0
        for dim in weighted_vector.keys():  # this is faster than list comprehension
            vector_sum += weighted_vector[dim] * unweighted_vector.get(dim, 0)
        for dim in set(unweighted_vector.keys()).difference(weighted_vector.keys()):
            vector_sum += weights[dim]
        return vector_sum

    @staticmethod
    def _norm(tokens):
        dimensions = defaultdict(int)
        for token in tokens:
            dimensions[token] += 1
        return sqrt(sum([d**2 for d in dimensions.values()]))
