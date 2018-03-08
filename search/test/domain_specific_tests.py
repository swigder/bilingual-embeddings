from baseline import CosineSimilaritySearchEngine
from search_engine import EmbeddingSearchEngine
from .run_tests import query_result, f1_score
from .testing_framework import EmbeddingsTest
from text_tools import tokenize, normalize


'''
OOV test - how many OOV terms in the queries wrt document collection or embeddings.
'''

oov_columns = ['tokens', 'tokens-oov', 'unique', 'unique-oov', 'examples']


def oov_rate(iv, oov):
    return len(oov) / (len(iv) + len(oov))


def texts_to_tokens(texts):
    tokens = []
    for text in texts:
        tokens += tokenize(normalize(text))
    return tokens


def oov_details(tokens, vocabulary):
    in_vocabulary = []
    out_of_vocabulary = []
    for token in tokens:
        in_vocabulary.append(token) if token in vocabulary else out_of_vocabulary.append(token)
    in_vocabulary_set, out_of_vocabulary_set = set(in_vocabulary), set(out_of_vocabulary)
    return {'tokens': len(out_of_vocabulary), 'tokens-oov': oov_rate(in_vocabulary, out_of_vocabulary),
            'unique': len(out_of_vocabulary_set), 'unique-oov': oov_rate(in_vocabulary_set, out_of_vocabulary_set), }
    # 'examples': list(out_of_vocabulary_set)[:10]}


def oov_test_f(collection, embed):
    query_tokens = texts_to_tokens(collection.queries.values())

    if embed is None:
        document_tokens = texts_to_tokens(collection.documents.values())
        return oov_details(query_tokens, set(document_tokens))
    else:
        return oov_details(tokens=query_tokens, vocabulary=embed)


oov_test = EmbeddingsTest(f=oov_test_f, columns=oov_columns, non_embeddings='documents')


'''
Search test - precision / recall.
'''

search_test_columns = ['precision', 'recall', 'f-score']


def search_test_f(collection, embed):
    if embed:
        search_engine = EmbeddingSearchEngine(dictionary=embed)
    else:
        search_engine = CosineSimilaritySearchEngine()
    search_engine.index_documents(documents=collection.documents.values())

    total_precision, total_recall = 0, 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}
    for i, query in collection.queries.items():
        expected = collection.relevance[i]
        pr = query_result(search_engine, i, query, expected, doc_ids, 5, verbose=False)
        total_precision += pr.precision
        total_recall += pr.recall
    precision, recall = total_precision / len(collection.queries), total_recall / len(collection.queries)
    return {'precision': precision, 'recall': recall, 'f-score': f1_score(precision=precision, recall=recall)}


search_test = EmbeddingsTest(f=search_test_f, columns=search_test_columns, non_embeddings='baseline')


