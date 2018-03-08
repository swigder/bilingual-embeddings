import argparse

from .domain_specific_tests import oov_test, search_test
from ir_data_reader import readers, read_collection
from .testing_framework import vary_embeddings


parser = argparse.ArgumentParser(description='IR data reader.')
subparsers = parser.add_subparsers()

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('ir_dir', type=str, help='Directory with IR files', nargs='?')
parent_parser.add_argument('-t', '--types', choices=list(readers.keys()) + ['all'], nargs='*', default='all')

parent_parser.add_argument('-d', '--domain_embed', type=str, nargs='*',
                           help='Embedding format for domain-specific embedding')
parent_parser.add_argument('-e', '--embed', type=str, nargs='*',
                           help='Embedding location for general purpose embedding')

parser_domain_specific_oov = subparsers.add_parser('oov', parents=[parent_parser])
parser_domain_specific_oov.set_defaults(func=vary_embeddings(oov_test))

parser_domain_specific_search = subparsers.add_parser('search', parents=[parent_parser])
parser_domain_specific_search.set_defaults(func=vary_embeddings(search_test))

args = parser.parse_args()

if args.types == 'all':
    args.types = list(readers.keys())

args.func([read_collection(base_dir=args.ir_dir, collection_name=name) for name in args.types], args)
