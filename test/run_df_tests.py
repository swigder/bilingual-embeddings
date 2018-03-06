import argparse

import os

from math import log

import numpy as np
import pandas as pd

from search.baseline import CosineSimilaritySearchEngine
from search.dictionary import MonolingualDictionary
from utils.ir_data_reader import readers
from test.run_tests import f1_score, test_search_engine
from search import EmbeddingSearchEngine


def compare_df_options(baseline=True):
    tf_fns = {
        'plain': lambda tf: tf,
        'log': lambda tf: (1 + log(tf, 10)) if tf is not 0 else 0
    }
    df_cutoffs = [0.2, 0.4, 0.6, 0.8, 1.0]
    df_files = {'none': None, 'wiki': '/Users/xx/thesis/wiki-df/wiki-df-fasttext.txt'}
    default_df_fns = {
        'zero': lambda dfs: 0,
        'min': lambda dfs: np.min(dfs),
        'avg': lambda dfs: np.average(dfs),
    } if not baseline else {'zero': lambda dfs: 0}
    df2weights = {
        'none': lambda df, num_docs: 1,
        'plain': lambda df, num_docs: 1/df,
        'log': lambda df, num_docs: log(num_docs / df, 10),
    }

    df_option_options = [df_files.keys(), tf_fns.keys(), default_df_fns.keys(), df2weights.keys(), df_cutoffs]
    index = pd.MultiIndex.from_product(df_option_options, names=['df_file', 'tf', 'def_df', 'df', 'stopwords'])
    results = pd.DataFrame(index=index, columns=['precision', 'recall', 'f-score'])

    for key in index:
        df_file, tf, def_df, df, stopwords = key
        print('\n' + str(list(zip(index.names, key))))
        df_config = {'df_cutoff': stopwords,
                     'default_df_fn': default_df_fns[def_df],
                     'df_file': df_files[df_file],
                     'df_to_weight': df2weights[df]}
        tf_idf_options = {'tf_function': tf_fns[tf], 'df_options': df_config}
        if baseline:
            search_engine = CosineSimilaritySearchEngine(tf_idf_options=tf_idf_options)
        else:
            search_engine = EmbeddingSearchEngine(dictionary=mono_dict, tf_idf_options=tf_idf_options)
        search_engine.index_documents(ir_collection.documents.values())
        precision, recall = test_search_engine(search_engine, ir_collection, verbose=False)
        results.loc[key] = {'precision': precision, 'recall': recall}
    results['f-score'] = results.apply(lambda row: f1_score(row['precision'], row['recall']), axis=1)
    print('\n\nResults:\n')
    print(results.sort_values('f-score', ascending=False).to_string())


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
