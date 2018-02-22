import argparse
import operator

import os

import itertools

from math import log

import numpy as np

from baseline import CosineSimilaritySearchEngine
from dictionary import MonolingualDictionary
from ir_data_reader import readers
from run_tests import f1_score, test_search_engine
from search_engine import EmbeddingSearchEngine


def compare_df_options(baseline=True):
    tf_fns = {
        'plain': lambda tf: tf,
        'log': lambda tf: (1 + log(tf, 10)) if tf is not 0 else 0
    }
    df_cutoffs = [0.2, 0.4, 0.6, 0.8, 1.0]
    df_files = [None, '/Users/xx/thesis/wiki-df/wiki-df-fasttext.txt']
    default_df_fns = {
        'zero': lambda dfs: 0,
        'min': lambda dfs: np.min(dfs),
        'avg': lambda dfs: np.average(dfs),
    } if not baseline else {'zero': lambda dfs: 0}
    df2weights = {
        'plain': lambda df, num_docs: 1/df,
        'log': lambda df, num_docs: log(num_docs / df, 10),
    }
    df_option_options = [tf_fns.items(), df_cutoffs, df_files, default_df_fns.items(), df2weights.items()]
    results = {}
    for tf_fn, df_cutoff, df_file, default_df_fn, df2weight in itertools.product(*df_option_options):
        key = 'tf {} - df_file {} - stop {} - defdf {} - df2w {}'.format(
            tf_fn[0], df_file, df_cutoff, default_df_fn[0], df2weight[0])
        print('\n' + key)
        df_config = {'df_cutoff': df_cutoff,
                     'default_df_fn': default_df_fn[1],
                     'df_file': df_file,
                     'df_to_weight': df2weight[1]}
        if baseline:
            search_engine = CosineSimilaritySearchEngine(tf_idf_options={'tf_function': tf_fn[1], 'df_options': df_config})
        else:
            search_engine = EmbeddingSearchEngine(dictionary=mono_dict, df_file=df_file, df_options=df_config)
        search_engine.index_documents(ir_collection.documents.values())
        precision, recall = test_search_engine(search_engine, ir_collection, verbose=False)
        results[key] = f1_score(precision, recall)
    print('\n\nResults:\n')
    for k, v in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
        print('{} {:.4f}'.format(k, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('ir_dir', type=str, help='Directory with IR files', nargs='?')
    parser.add_argument('embed', type=str, help='Embedding file', nargs='?')
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')
    parser.add_argument('-n', '--number_results', type=int, default=5)
    parser.add_argument('-b', '--baseline', action='store_true')

    args = parser.parse_args()

    if not args.ir_dir:
        args.ir_dir = '/Users/xx/Documents/school/kth/thesis/ir-datasets/'
    if not args.embed:
        args.embed = '/Users/xx/thesis/bivecs-muse/wiki.multi.en.vec'

    reader = readers[args.type](os.path.join(args.ir_dir, args.type))
    ir_collection = reader.read_documents_queries_relevance()

    search_engines = {}

    if not args.baseline:
        mono_dict = MonolingualDictionary(emb_file=args.embed)

    compare_df_options(baseline=args.baseline)
