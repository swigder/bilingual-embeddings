import itertools

from math import log

import numpy as np
import pandas as pd

from baseline import CosineSimilaritySearchEngine
from dictionary import MonolingualDictionary
from search_engine import EmbeddingSearchEngine


def generate_dict_getter(non_domain_embed, domain_embed):
    assert len(non_domain_embed) + len(domain_embed) == 1

    if len(non_domain_embed) == 1:
        dictionary = MonolingualDictionary(non_domain_embed[0])
        return lambda collection: dictionary
    else:
        return lambda collection: MonolingualDictionary(domain_embed[0].format(collection.name))


tf_fns = {
    'plain': lambda tf: tf,
    'log': lambda tf: (1 + log(tf, 10)) if tf is not 0 else 0
}
default_df_fns = {
    'zero': lambda dfs: 0,
    'min': lambda dfs: np.min(dfs),  # min df = max weight
    'max': lambda dfs: np.max(dfs),  # min df = max weight
    'avg': lambda dfs: np.average(dfs),
}
df2weights = {
    'one': lambda df, num_docs: 1,
    'plain': lambda df, num_docs: 1 / df,
    'log': lambda df, num_docs: log(num_docs / df, 10),
}


def add_df_parser_options(parser):
    parser.add_argument('--tf_fns', choices=tf_fns.keys(), nargs='*', default=tf_fns.keys())
    parser.add_argument('--default_df_fns', choices=default_df_fns.keys(), nargs='*', default=default_df_fns.keys())
    parser.add_argument('--df2weights', choices=df2weights.keys(), nargs='*', default=df2weights.keys())
    parser.add_argument('--stopword_cutoffs', type=float, nargs='*', default=[.4, .8])
    parser.add_argument('--df_files', type=str, nargs='*', default=[''])
    parser.add_argument('--non_grid', type=str, nargs='*', default=None)


def df_config_generator(parsed_args):
    if not parsed_args.non_grid:
        df_option_options = [parsed_args.tf_fns,
                             parsed_args.stopword_cutoffs,
                             parsed_args.df_files,
                             parsed_args.default_df_fns,
                             parsed_args.df2weights]
        for tf_fn, stopword_cutoff, df_file, default_df_fn, df2weight in itertools.product(*df_option_options):
            key = (tf_fn, stopword_cutoff, df_file, default_df_fn, df2weight)
            value = {'df_cutoff': stopword_cutoff,
                     'default_df_fn': default_df_fns[default_df_fn],
                     'df_file': df_file or None,
                     'df_to_weight': df2weights[df2weight]}
            yield key, value
    else:
        for options in parsed_args.non_grid:
            tf_fn, stopword_cutoff, default_df_fn, df_file, df2weight = options.split()
            key = (tf_fn, stopword_cutoff, df_file, default_df_fn, df2weight)
            value = {'df_cutoff': float(stopword_cutoff),
                     'default_df_fn': default_df_fns[default_df_fn],
                     'df_file': df_file if df_file is not 'x' else None,
                     'df_to_weight': df2weights[df2weight]}
            yield key, value


def vary_df(test):
    def inner(collections, parsed_args):

        baseline = test.non_embed and parsed_args.baseline
        columns = [x[0] for x in df_config_generator(parsed_args)]
        df = pd.DataFrame(index=(c.name for c in collections), columns=columns)

        dict_getter = None if baseline else generate_dict_getter(non_domain_embed=parsed_args.embed or [],
                                                                 domain_embed=parsed_args.domain_embed or [])

        for collection in collections:
            dictionary = None if baseline else dict_getter(collection)
            df_configs = df_config_generator(parsed_args)
            for key, df_config in df_configs:
                if baseline:
                    search_engine = CosineSimilaritySearchEngine(
                        tf_idf_options={'tf_function': None, 'df_options': df_config})
                else:
                    search_engine = EmbeddingSearchEngine(dictionary=dictionary, df_options=df_config)
                search_engine.index_documents(collection.documents.values())
                df.loc[collection.name, key] = test.f(collection, search_engine)

        return df

    return inner
