from .testing_framework import EmbeddingsTest
from text_tools import tokenize, normalize


'''
OOV test - how many OOV terms in the queries wrt document collection or embeddings.
'''

oov_columns = ['tokens-count', 'tokens-rate', 'type-count', 'type-rate']


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
    return {'tokens-count': len(out_of_vocabulary),
            'tokens-rate': oov_rate(in_vocabulary, out_of_vocabulary),
            'type-count': len(out_of_vocabulary_set),
            'type-rate': oov_rate(in_vocabulary_set, out_of_vocabulary_set), }
    # 'examples': list(out_of_vocabulary_set)[:10]}


def oov_test_f(collection, embed):
    query_tokens = texts_to_tokens(collection.queries.values())

    if embed is None:
        document_tokens = texts_to_tokens(collection.documents.values())
        return oov_details(tokens=query_tokens, vocabulary=set(document_tokens))
    else:
        return oov_details(tokens=query_tokens, vocabulary=embed)


oov_test = EmbeddingsTest(f=oov_test_f, columns=oov_columns, non_embed='documents')


