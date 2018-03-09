import os
from collections import namedtuple

import pandas as pd

from dictionary import MonolingualDictionary


EmbeddingsTest = namedtuple('EmbeddingsTest', ['f', 'non_embeddings', 'columns'])


def vary_embeddings(test):
    base_name_map = lambda ps: {os.path.basename(p): p for p in ps or []}

    def inner(collections, parsed_args):
        non_domain_embed = base_name_map(parsed_args.embed)
        domain_embed = base_name_map(parsed_args.domain_embed)

        baseline = test.non_embeddings and parsed_args.baseline
        index = pd.MultiIndex.from_product([(c.name for c in collections),
                                            ([test.non_embeddings] if baseline else []) +
                                            list(non_domain_embed.keys()) + list(domain_embed.keys())])
        df = pd.DataFrame(index=index, columns=test.columns)

        # embeddings are slow to load and take up a lot of memory. load them only once for all collections, and release
        # them quickly.
        for embed_name, path in non_domain_embed.items():
            embed = MonolingualDictionary(path)
            for collection in collections:
                df.loc[collection.name, embed_name] = test.f(collection, embed)

        for collection in collections:
            if baseline:
                df.loc[collection.name, test.non_embeddings] = test.f(collection, None)
            for embed_name, path in domain_embed.items():
                embed = MonolingualDictionary(path.format(collection.name))
                df.loc[collection.name, embed_name] = test.f(collection, embed)

        print(df)

    return inner


def split_types(f):
    return lambda cs, a: (f(c, a) for c in cs)
