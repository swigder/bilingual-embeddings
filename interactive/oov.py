import matplotlib.pyplot as plt
import pandas as pd

from test.__main__ import collections, args
from test.oov_tests import oov_test
from test.testing_framework import vary_embeddings, multirun_map, embed_to_engine, search_test


def get_df():
    args.embed = ['/Users/xx/thesis/embed/crawl.vec',
                  '/Users/xx/thesis/embed/wiki.vec',
                  '/Users/xx/thesis/embed/pubmed.bin',
                  ]
    args.domain_embed = ['/Users/xx/thesis/embed/baselines/{}-only-sub-3-win-20.vec',
                         '/Users/xx/thesis/embed/baselines/{}-crawl-sub-3-win-20.vec',
                         '/Users/xx/thesis/embed/baselines/{}-wiki-sub-3-win-20.vec',
                         ]

    args.func = vary_embeddings(oov_test)
    oov_result = multirun_map(args.func)(collections, args)

    args.func = vary_embeddings(embed_to_engine(search_test))
    map_result = multirun_map(args.func)(collections, args)

    df = pd.merge(oov_result, map_result, left_index=True, right_index=True)
    df = df.drop(['tokens-count', 'unique-count', 'unique-rate'], axis=1)
    df = df.astype('float64')

    return df


def plot():
    plt.scatter(df.loc['adi', 'tokens-rate'], df.loc['adi', 'MAP@10'], label='adi', marker='o')
    plt.scatter(df.loc['time', 'tokens-rate'], df.loc['time', 'MAP@10'], label='time', marker='x')
    plt.scatter(df.loc['ohsu-trec', 'tokens-rate'], df.loc['ohsu-trec', 'MAP@10'], label='ohsu-trec', marker='^')
    plt.legend(frameon=True).get_frame().set_facecolor('white')
    plt.xlabel('OOV rate')
    plt.ylabel('MAP@10')
    plt.show()


df = get_df()
plot()
