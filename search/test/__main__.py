import argparse

from .df_tests import vary_df, add_df_parser_options
from .oov_tests import oov_test
from .testing_framework import vary_embeddings, search_test, embed_to_engine, display_data
from ir_data_reader import readers, read_collection


parser = argparse.ArgumentParser(description='IR data reader.')
subparsers = parser.add_subparsers()

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('ir_dir', type=str, help='Directory with IR files')
parent_parser.add_argument('-t', '--types', choices=list(readers.keys()) + ['all'], nargs='*', default='all')
parent_parser.add_argument('-b', '--baseline', action='store_true')
parent_parser.add_argument('-s', '--subword', action='store_true')
parent_parser.add_argument('-hp', '--hyperparams', action='store_true')
parent_parser.add_argument('-c', '--column', type=str, nargs='?')
parent_parser.add_argument('-l', '--latex', action='store_true')
parent_parser.add_argument('-p', '--precision', type=int, default=4)
parent_parser.add_argument('-q', '--query_id', type=str, nargs='*')

parent_parser.add_argument('-d', '--domain_embed', type=str, nargs='*',
                           help='Embedding format for domain-specific embedding')
parent_parser.add_argument('-e', '--embed', type=str, nargs='*',
                           help='Embedding location for general purpose embedding')

oov_parser = subparsers.add_parser('oov', parents=[parent_parser])
oov_parser.set_defaults(func=vary_embeddings(oov_test))

embedding_search_parser = subparsers.add_parser('embed', parents=[parent_parser])
embedding_search_parser.set_defaults(func=vary_embeddings(embed_to_engine(search_test)))

df_parser = subparsers.add_parser('df', parents=[parent_parser])
add_df_parser_options(df_parser)
df_parser.set_defaults(func=vary_df(search_test))


args = parser.parse_args()

if args.types == 'all':
    args.types = list(readers.keys())

result = args.func([read_collection(base_dir=args.ir_dir, collection_name=name) for name in args.types], args)

display_data(result, args)
