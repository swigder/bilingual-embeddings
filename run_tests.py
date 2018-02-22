import argparse
import itertools
import operator
import os

from collections import namedtuple

from math import sqrt

import numpy as np

from dictionary import MonolingualDictionary
from ir_data_reader import readers, sub_collection
from search_engine import EmbeddingSearchEngine
from baseline import CosineSimilaritySearchEngine


PrecisionRecall = namedtuple('PrecisionRecall', ['precision', 'recall'])


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def precision_recall(expected, actual):
    true_positives = sum([1 for item in expected if item in actual])
    return true_positives / len(actual), true_positives / len(expected)


def read_data(data_dir, doc_file, query_file, relevance_file, reader):
    f = lambda file: os.path.join(data_dir, file)

    return reader.read(f(doc_file), f(query_file), f(relevance_file))


def query_result(search_engine, i, query, expected, documents, n=5, verbose=False):
    results = search_engine.query_index(query, n_results=n)
    results_i = []
    if verbose:
        print()
        print(i, query, expected)
    for distance, result in results:
        result_i = documents[result]
        results_i.append(result_i)
        if verbose:
            print(result_i, distance, result[:300])
            if result_i in expected:
                print('Correct!')
    precision, recall = precision_recall(expected=expected, actual=results_i)
    return PrecisionRecall(precision, recall)


def compare_search_engines(search_engine, baseline, collection, n=5, print_details=True):
    total_precision, total_recall = 0, 0
    base_precision, base_recall = 0, 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}

    for i, query in collection.queries.items():
        expected = collection.relevance[i]

        engine_pr = query_result(search_engine, i, query, expected, doc_ids, n, verbose=False)
        base_pr = query_result(baseline, i, query, expected, doc_ids, n, verbose=False)

        total_precision += engine_pr.precision
        total_recall += engine_pr.recall
        base_precision += base_pr.precision
        base_recall += base_pr.recall

        if print_details and engine_pr.precision < base_pr.precision or engine_pr.recall < base_pr.recall:
            query_result(search_engine, i, query, expected, doc_ids, n, verbose=True)
            print('System: {:.4f} / {:.4f}'.format(engine_pr.precision, engine_pr.recall))
            query_result(baseline, i, query, expected, doc_ids, n, verbose=True)
            print('Baseline: {:.4f} / {:.4f}'.format(base_pr.precision, base_pr.recall))

    print()
    print('         Precision / Recall: {:.4f} / {:.4f}'.format(total_precision / len(collection.queries),
                                                                total_recall / len(collection.queries)))
    print('Baseline Precision / Recall: {:.4f} / {:.4f}'.format(base_precision / len(collection.queries),
                                                                base_recall / len(collection.queries)))


def test_search_engine(search_engine, collection, n=5, verbose=False):
    total_precision, total_recall = 0, 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}
    for i, query in collection.queries.items():
        expected = collection.relevance[i]
        pr = query_result(search_engine, i, query, expected, doc_ids, n, verbose=verbose)
        total_precision += pr.precision
        total_recall += pr.recall
    if verbose:
        print()
    precision, recall = total_precision / len(collection.queries), total_recall / len(collection.queries)
    # print('Precision / Recall: {:.4f} / {:.4f}'.format(precision, recall))
    return precision, recall


def compare_df_options():
    df_cutoffs = [.8]
    smoothing_fns = {
        'none': lambda num_docs: 0,
        'one': lambda num_docs: 1,
        'sqrt': lambda num_docs: int(sqrt(num_docs)),
    }
    default_df_fns = {
        'zero': lambda dfs, smoothing: 0,
        'min': lambda dfs, smoothing: np.min(dfs),
        'avg': lambda dfs, smoothing: np.average(dfs),
        'avg-minus-smoothing': lambda dfs, smoothing: np.average(dfs) - smoothing,
        'smoothing': lambda dfs, smoothing: max(smoothing, 1)
    }
    bucket_options = [None, 5, 10, 100]
    df_option_options = [df_cutoffs, smoothing_fns.items(), default_df_fns.items(), bucket_options]
    results = {}
    for df_file in [None, '/Users/xx/thesis/wiki-df/wiki-df-fasttext.txt']:
        for current_options in itertools.product(*df_option_options):
            df_cutoff, smoothing_fn, default_df_fn, buckets = current_options
            if smoothing_fn[0] == 'none' and default_df_fn[0] == 'avg-minus-smoothing':
                continue
            if df_file is not None and buckets is None:
                continue
            df_config = {'df_cutoff': df_cutoff,
                         'smoothing_fn': smoothing_fn[1],
                         'default_df_fn': default_df_fn[1],
                         'buckets': buckets}
            search_engine = EmbeddingSearchEngine(dictionary=mono_dict, df_file=df_file, df_options=df_config)
            search_engine.index_documents(ir_collection.documents.values())
            precision, recall = test_search_engine(search_engine, ir_collection, verbose=False)
            key = df_file, df_cutoff, smoothing_fn[0], default_df_fn[0], buckets
            results[key] = f1_score(precision, recall)
    for k, v in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
        print('{} {:.4f}'.format(k, v))


def sub_query(query_i, n=None, search_engine=None):
    n = n if n else args.number_results
    search_engine = search_engine if search_engine else list(search_engines.values())[0]
    query_collection = sub_collection(ir_collection, query_i)
    test_search_engine(search_engine, query_collection, n=n, verbose=True)


def print_df(search_engine, query_i):
    from nltk import word_tokenize
    from operator import itemgetter
    dfs = {token: search_engine.df[token] for token in word_tokenize(ir_collection.queries[query_i])}
    for token, df in sorted(dfs.items(), key=itemgetter(1)):
        print(token, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('ir_dir', type=str, help='Directory with IR files', nargs='?')
    parser.add_argument('embed', type=str, help='Embedding file', nargs='?')
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')
    parser.add_argument('-n', '--number_results', type=int, default=5)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-b', '--baseline', action='store_true')
    parser.add_argument('-c', '--compare', action='store_true')
    parser.add_argument('-d', '--df_file', type=str, help='File with precomputed df', default=None)

    args = parser.parse_args()

    if not args.ir_dir:
        args.ir_dir = '/Users/xx/Documents/school/kth/thesis/ir-datasets/'
    if not args.embed:
        args.embed = '/Users/xx/thesis/bivecs-muse/wiki.multi.en.vec'
        # args.embed = '/Users/xx/Downloads/GoogleNews-vectors-negative300.bin.gz'

    reader = readers[args.type](os.path.join(args.ir_dir, args.type))
    ir_collection = reader.read_documents_queries_relevance()

    search_engines = {}

    if args.compare or not args.baseline:
        mono_dict = MonolingualDictionary(emb_file=args.embed)
        search_engines['Embedding'] = EmbeddingSearchEngine(dictionary=mono_dict, df_file=args.df_file)
    if args.compare or args.baseline:
        search_engines['Baseline'] = CosineSimilaritySearchEngine()

    # for engine in search_engines.values():
    #     engine.index_documents(ir_collection.documents.values())
    #
    # if args.compare:
    #     compare_search_engines(search_engines['Embedding'], search_engines['Baseline'],
    #                            ir_collection, n=args.number_results)
    # else:
    #     engine = list(search_engines.values())[0]
    #     test_search_engine(engine, ir_collection, n=args.number_results, verbose=args.verbose)

    compare_df_options()
