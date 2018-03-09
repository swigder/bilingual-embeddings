import os
from collections import namedtuple

import pandas as pd

from baseline import CosineSimilaritySearchEngine
from dictionary import MonolingualDictionary
from search_engine import EmbeddingSearchEngine
from .run_tests import query_result, f1_score


EmbeddingsTest = namedtuple('EmbeddingsTest', ['f', 'non_embed', 'columns'])


def indexed_search_engine(embed, collection):
    if embed:
        search_engine = EmbeddingSearchEngine(dictionary=embed)
    else:
        search_engine = CosineSimilaritySearchEngine()
    search_engine.index_documents(documents=collection.documents.values())
    return search_engine


def vary_embeddings(test):
    base_name_map = lambda ps: {os.path.basename(p): p for p in ps or []}

    def inner(collections, parsed_args):
        non_domain_embed = base_name_map(parsed_args.embed)
        domain_embed = base_name_map(parsed_args.domain_embed)

        baseline = test.non_embed and parsed_args.baseline
        index = pd.MultiIndex.from_product([(c.name for c in collections),
                                            ([test.non_embed] if baseline else []) +
                                            list(non_domain_embed.keys()) + list(domain_embed.keys())])
        df = pd.DataFrame(index=index, columns=test.columns)

        # embeddings are slow to load and take up a lot of memory. load them only once for all collections, and release
        # them quickly.
        for embed_name, path in non_domain_embed.items():
            embed = MonolingualDictionary(path)
            for collection in collections:
                df.loc[collection.name, embed_name] = test.f(collection, indexed_search_engine(embed, collection))

        for collection in collections:
            if baseline:
                df.loc[collection.name, test.non_embed] = test.f(collection, indexed_search_engine(None, collection))
            for embed_name, path in domain_embed.items():
                embed = MonolingualDictionary(path.format(collection.name))
                df.loc[collection.name, embed_name] = test.f(collection, indexed_search_engine(embed, collection))

        print(df)

    return inner


def split_types(f):
    return lambda cs, a: (f(c, a) for c in cs)


'''
Search test - precision / recall.
'''

search_test_columns = ['precision', 'recall', 'f-score']


def search_test_f(collection, search_engine):
    total_precision, total_recall = 0, 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}
    for i, query in collection.queries.items():
        expected = collection.relevance[i]
        pr = query_result(search_engine, i, query, expected, doc_ids, 5, verbose=False)
        total_precision += pr.precision
        total_recall += pr.recall
    precision, recall = total_precision / len(collection.queries), total_recall / len(collection.queries)
    return {'precision': precision, 'recall': recall, 'f-score': f1_score(precision=precision, recall=recall)}


search_test = EmbeddingsTest(f=search_test_f, columns=search_test_columns, non_embed='baseline')

