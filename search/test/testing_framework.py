import copy
import glob
import os
from collections import namedtuple

import pandas as pd
from numpy import average

from baseline import CosineSimilaritySearchEngine
from dictionary import dictionary
from search_engine import EmbeddingSearchEngine, BilingualEmbeddingSearchEngine
from utils import print_with_time
from .run_tests import query_result, f1_score, average_precision

EmbeddingsTest = namedtuple('EmbeddingsTest', ['f', 'non_embed', 'columns'])

base_name_map = lambda ps: {os.path.splitext(os.path.basename(p))[0].replace('{}-', 'Coll+'): p for p in ps or []}
df_value_gen = lambda parsed_args: lambda value: value if not parsed_args.column else value[parsed_args.column]


def multirun_map(test):
    def inner(collections, parsed_args):
        def run_with_base(base):
            def add_dir(path):
                head, tail = os.path.split(path)
                return os.path.join(base, tail) if not head else path
            updated_parsed_args = copy.copy(parsed_args)
            updated_parsed_args.embed = list(map(add_dir, parsed_args.embed or []))
            updated_parsed_args.domain_embed = list(map(add_dir, parsed_args.domain_embed or []))
            return test(collections, updated_parsed_args)

        if not parsed_args.embed_location:
            return test(collections, parsed_args)
        if not parsed_args.multirun:
            return run_with_base(parsed_args.embed_location)
        results = []
        for base in filter(os.path.isdir, glob.glob(os.path.join(parsed_args.embed_location, '*'))):
            results.append(run_with_base(base))
        return results
    return inner


def hyperparameters(test):
    def inner(collections, parsed_args):
        df = pd.DataFrame(index=[c.name for c in collections], columns=[])
        df_value = df_value_gen(parsed_args)

        for collection in collections:
            paths = list(parsed_args.embed)
            for domain_embed_path in parsed_args.domain_embed:
                paths.append(domain_embed_path.format(collection.name))
            for globbed_path in paths:
                embeds = glob.glob(globbed_path)
                for embed_path in embeds:
                    embed = dictionary(embed_path, use_subword=parsed_args.subword, normalize=parsed_args.normalize)
                    star = globbed_path.index('*')
                    column = embed_path[star:star-len(globbed_path)+1]
                    df.loc[collection.name, column] = df_value(test.f(collection, embed))
            if parsed_args.relative:
                baseline = df.loc[collection.name, parsed_args.relative]
                df.loc[collection.name] = ((df.loc[collection.name] / baseline) - 1) * 100

        return df

    return inner


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
    def inner(collections, parsed_args):
        df_value = df_value_gen(parsed_args)

        # use base name as prettier format, None -> []
        non_domain_embed = base_name_map(parsed_args.embed)
        domain_embed = base_name_map(parsed_args.domain_embed)

        # embeddings are slow to load. fail fast if one doesn't exist.
        for path in non_domain_embed.values():
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        for path in domain_embed.values():
            for collection in collections:
                if not os.path.exists(path.format(collection.name)):
                    raise FileNotFoundError(path.format(collection.name))
        if len(collections) == 1 and len(domain_embed.values()) == 1:
            only_path = list(domain_embed.values())[0]
            if os.path.isdir(only_path):
                paths = glob.glob(os.path.join(only_path, '{}*.bin'.format(collections[0].name)))
                if len(paths) == 0:
                    paths = glob.glob(os.path.join(only_path, '{}*.vec'.format(collections[0].name)))
                domain_embed = {path: path for path in paths}
                print_with_time('Found {} embeddings to test.'.format(len(domain_embed)))

        baseline = test.non_embed and parsed_args.baseline
        embed_names = [test.non_embed] if baseline else [] + list(non_domain_embed.keys()) + list(domain_embed.keys())
        if parsed_args.column:
            index = [c.name for c in collections]
            columns = embed_names
        else:
            index = pd.MultiIndex.from_product([(c.name for c in collections), embed_names])
            columns = test.columns
        df = pd.DataFrame(index=index, columns=columns)

        # embeddings are slow to load and take up a lot of memory. load them only once for all collections, and release
        # them quickly.
        for embed_name, path in non_domain_embed.items():
            embed = dictionary(path, use_subword=parsed_args.subword, normalize=parsed_args.normalize)
            for collection in collections:
                df.loc[collection.name, embed_name] = df_value(test.f(collection, embed))

        total = len(collections) * len(domain_embed.items())
        for i, collection in enumerate(collections):
            if baseline:
                df.loc[collection.name, test.non_embed] = df_value(test.f(collection, None))
            for j, (embed_name, path) in enumerate(domain_embed.items()):
                embed = dictionary(path.format(collection.name),
                                   use_subword=parsed_args.subword, normalize=parsed_args.normalize)
                print_with_time('Testing ({}/{}) {}'.format(i*j+j+1, total, embed_name))
                df.loc[collection.name, embed_name] = df_value(test.f(collection, embed))
        return df

    return inner


def split_collections(f):
    return lambda cs, a: (f(c, a) for c in cs)


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
    queries = collection.queries.items() if type(search_engine) is not BilingualEmbeddingSearchEngine \
        else collection.queries_translated.items()
    for i, query in queries:
        expected = collection.relevance[i]
        total_average_precision += query_result(search_engine, i, query, expected, doc_ids, 10,
                                                verbose=False,
                                                metric=average_precision)
    return total_average_precision / len(queries)


search_test = EmbeddingsTest(f=search_test_map, columns=['MAP@10'], non_embed='baseline')


def recall_test_f(collection, search_engine):
    max_distances = []
    max_ranks = []
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}
    for i, query in collection.queries.items():
        expected = collection.relevance[i]
        # for n in range(100, len(doc_ids) + 100, 100):
        #     results = search_engine.query_index(query, n_results=n)
        #     result_ids = [doc_ids[result[1]] for result in results]
        #     if not set(expected).issubset(result_ids):
        #         continue
        #     max_rank = max([result_ids.index(expected_i) for expected_i in expected])
        #     max_distances.append(results[max_rank][0])
        #     max_ranks.append(max_rank)
        #     break
        results = search_engine.query_index(query, n_results=len(doc_ids))
        max_ranks.append(0)
        max_distances.append(results[-1][0])
    assert len(max_distances) == len(max_ranks) == len(collection.queries)
    return {'Max dist (avg)': average(max_distances),
            'Max dist (max)': max(max_distances),
            'Max rank (avg)': average(max_ranks),
            'Max rank (max)': max(max_ranks),}


recall_test = EmbeddingsTest(f=recall_test_f,
                             columns=['Max dist (avg)', 'Max dist (max)', 'Max rank (avg)', 'Max rank (max)'],
                             non_embed='baseline')
