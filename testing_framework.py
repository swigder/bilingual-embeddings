import argparse

import os

import pandas as pd

from dictionary import MonolingualDictionary
from domain_specific_tests import oov_test
from ir_data_reader import readers, read_collection


def vary_embeddings(test):
    base_name_map = lambda ps: {os.path.basename(p): p for p in ps or []}

    def inner(collections, parsed_args):
        non_domain_embed = base_name_map(parsed_args.embed)
        domain_embed = base_name_map(parsed_args.domain_embed)

        index = pd.MultiIndex.from_product([(c.name for c in collections),
                                            ([test.non_embeddings] if test.non_embeddings else []) +
                                            list(non_domain_embed.keys()) + list(domain_embed.keys())])
        df = pd.DataFrame(index=index, columns=test.columns)

        non_domain_embed = {n: MonolingualDictionary(p) for n, p in non_domain_embed.items()}

        for collection in collections:
            if test.non_embeddings:
                test.f(df, collection, None, test.non_embeddings)
            for name, embed in non_domain_embed.items():
                test.f(df, collection, embed, name)
            for name, path in domain_embed.items():
                embed = MonolingualDictionary(path.format(collection.name))
                test.f(df, collection, embed, name)

        print(df)

    return inner


def split_types(f):
    return lambda cs, a: (f(c, a) for c in cs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')
    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('ir_dir', type=str, help='Directory with IR files', nargs='?')
    parent_parser.add_argument('-t', '--types', choices=list(readers.keys()) + ['all'], nargs='*', default='all')

    parent_parser.add_argument('-d', '--domain_embed', type=str, nargs='*',
                               help='Embedding format for domain-specific embedding')
    parent_parser.add_argument('-e', '--embed', type=str, nargs='*',
                               help='Embedding location for general purpose embedding')

    parser_fasttext = subparsers.add_parser('oov', parents=[parent_parser])
    parser_fasttext.set_defaults(func=vary_embeddings(oov_test))

    args = parser.parse_args()

    if args.types == 'all':
        args.types = list(readers.keys())

    args.func([read_collection(base_dir=args.ir_dir, collection_name=name) for name in args.types], args)
