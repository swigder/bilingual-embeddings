import glob
import os
from collections import namedtuple

import pandas as pd
import matplotlib.pyplot as plt

from baseline import CosineSimilaritySearchEngine
from dictionary import MonolingualDictionary, SubwordDictionary, BilingualDictionary
from search_engine import EmbeddingSearchEngine, BilingualEmbeddingSearchEngine
from .run_tests import query_result, f1_score, average_precision

EmbeddingsTest = namedtuple('EmbeddingsTest', ['f', 'non_embed', 'columns'])


def embed_to_engine(test):
    def inner(collection, embed):
        if embed:
            search_engine = EmbeddingSearchEngine(dictionary=embed)
        else:
            search_engine = CosineSimilaritySearchEngine()
        search_engine.index_documents(documents=collection.documents.values())
        return test.f(collection, search_engine)
    return EmbeddingsTest(f=inner, non_embed=test.non_embed, columns=test.columns)


def vary_embeddings(test):
    base_name_map = lambda ps: {os.path.splitext(os.path.basename(p))[0].replace('{}-', 'Coll+'): p for p in ps or []}

    def inner(collections, parsed_args):
        def df_value(value):
            return value if not parsed_args.column else value[parsed_args.column]

        def dictionary(embed_path):
            return SubwordDictionary(embed_path) if parsed_args.subword else MonolingualDictionary(embed_path)

        # use base name as prettier format, None -> []
        non_domain_embed = base_name_map(parsed_args.embed)
        domain_embed = base_name_map(parsed_args.domain_embed)

        # embeddings are slow to load. fail fast if one doesn't exist.
        for path in non_domain_embed.values():
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        if not parsed_args.hyperparams:
            for path in domain_embed.values():
                for collection in collections:
                    if not os.path.exists(path.format(collection.name)):
                        raise FileNotFoundError(path.format(collection.name))

        baseline = test.non_embed and parsed_args.baseline
        embed_names = [test.non_embed] if baseline else [] + list(non_domain_embed.keys()) + list(domain_embed.keys())
        if not parsed_args.column:
            index = pd.MultiIndex.from_product([(c.name for c in collections), embed_names])
            columns = test.columns
        else:
            assert parsed_args.column in test.columns
            index = [c.name for c in collections]
            columns = embed_names if not parsed_args.hyperparams else []
        df = pd.DataFrame(index=index, columns=columns)

        # embeddings are slow to load and take up a lot of memory. load them only once for all collections, and release
        # them quickly.
        for embed_name, path in non_domain_embed.items():
            embed = dictionary(path)
            for collection in collections:
                df.loc[collection.name, embed_name] = df_value(test.f(collection, embed))

        for collection in collections:
            if baseline:
                df.loc[collection.name, test.non_embed] = df_value(test.f(collection, None))
            for embed_name, path in domain_embed.items():
                if not parsed_args.hyperparams:
                    embed = dictionary(path.format(collection.name))
                    df.loc[collection.name, embed_name] = df_value(test.f(collection, embed))
                else:
                    globbed_path = path.format(collection.name)
                    embeds = glob.glob(globbed_path)
                    for embed_path in embeds:
                        embed = dictionary(embed_path)
                        star = globbed_path.index('*')
                        column = embed_path[star:star-len(globbed_path)+1]
                        df.loc[collection.name, column] = df_value(test.f(collection, embed))
        if parsed_args.hyperparams:
            cols = df.columns.tolist()
            try:
                cols = list(map(str, sorted(map(int, cols))))
            except ValueError:
                cols = sorted(cols)
            df = df[cols]
        return df

    return inner


def split_collections(f):
    return lambda cs, a: (f(c, a) for c in cs)


def bilingual(test):
    def inner(collections, parsed_args):
        if len(collections) != 1:
            raise ValueError
        collection = collections[0]
        dictionary = BilingualDictionary(parsed_args.doc_embed, parsed_args.query_embed)
        search_engine = BilingualEmbeddingSearchEngine(dictionary=dictionary)
        search_engine.index_documents(collection.documents.values())
        doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}

        total_precision_original, total_recall_original = 0, 0
        total_precision_translated, total_recall_translated = 0, 0

        for i, query in collection.queries_translated.items():
            expected = collection.relevance[i]
            print('\nOriginal:')
            pr_original = query_result(search_engine, i, collection.queries[i], expected, doc_ids, 5, verbose=True)
            print('\nTranslated:')
            pr_translated = query_result(search_engine, i, query, expected, doc_ids, 5, verbose=True)
            print('\n-- P/r original: {}, p/r translated: {}'.format(pr_original, pr_translated))
            total_precision_original += pr_original.precision
            total_recall_original += pr_original.recall
            total_precision_translated += pr_translated.precision
            total_recall_translated += pr_translated.recall

        count = len(collection.queries_translated)

        print('\n-- Totals:')
        print('-- P/r original: {:.2f} {:.2f}, p/r translated: {:.2f} {:.2f}'.format(total_precision_original / count,
                                                                                     total_recall_original / count,
                                                                                     total_precision_translated / count,
                                                                                     total_recall_translated / count))
    return inner


'''
Search test - precision / recall.
'''

search_test_columns = ['precision', 'recall', 'f-score']


def search_test_pr(collection, search_engine):
    total_precision, total_recall = 0, 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}
    for i, query in collection.queries.items():
        expected = collection.relevance[i]
        pr = query_result(search_engine, i, query, expected, doc_ids, 5, verbose=False)
        total_precision += pr.precision
        total_recall += pr.recall
    precision, recall = total_precision / len(collection.queries), total_recall / len(collection.queries)
    return {'precision': precision, 'recall': recall, 'f-score': f1_score(precision=precision, recall=recall)}


def search_test_map(collection, search_engine):
    total_average_precision = 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}
    for i, query in collection.queries.items():
        expected = collection.relevance[i]
        total_average_precision += query_result(search_engine, i, query, expected, doc_ids, 10, verbose=False,
                                                metric=average_precision)
    return {'map': total_average_precision / len(collection.queries)}


search_test = EmbeddingsTest(f=search_test_map, columns=['map'], non_embed='baseline')


'''
Print result
'''


def print_table(data, args):
    if args.column_order:
        try:
            data = data[data.columns[list(map(int, list(args.column_order)))]]
        except:
            print('Unable to rearrange columns!')  # don't want to fail just cuz we can't rearrange columns
    pd.set_option('precision', args.precision)
    if args.latex:
        print(data.to_latex())
    else:
        print(data)


def display_chart(data, args):
    for row in data.index:
        plt.plot(data.loc[row], label=row)
    plt.legend()
    plt.xlabel(args.x_axis)
    plt.ylabel(args.column)
    plt.show()
