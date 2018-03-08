from collections import namedtuple

from text_tools import tokenize, normalize


EmbeddingsTest = namedtuple('EmbeddingsTest', ['f', 'non_embeddings', 'columns'])


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


def oov_test_f(df, collection, embed, embed_name):
    query_tokens = texts_to_tokens(collection.queries.values())

    if embed is None:
        document_tokens = texts_to_tokens(collection.documents.values())
        df.loc[collection.name, embed_name] = oov_details(query_tokens, set(document_tokens))
    else:
        df.loc[collection.name, embed_name] = oov_details(tokens=query_tokens, vocabulary=embed)


oov_test = EmbeddingsTest(f=oov_test_f, columns=oov_columns, non_embeddings='documents')
