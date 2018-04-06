import argparse

from .df_tests import vary_df, add_df_parser_options
from .oov_tests import oov_test
from .testing_framework import vary_embeddings, search_test, embed_to_engine, print_table, display_chart, bilingual
from ir_data_reader import readers, read_collection


parser = argparse.ArgumentParser(description='Run tests on search engine.')
subparsers = parser.add_subparsers(title='subcommands')

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('ir_dir', type=str, help='Directory with IR files')
parent_parser.add_argument('-c', '--collections', choices=list(readers.keys()) + ['all'], nargs='*', default='all')
parent_parser.add_argument('-b', '--baseline', action='store_true')
parent_parser.add_argument('-s', '--subword', action='store_true')
parent_parser.add_argument('-hp', '--hyperparams', action='store_true')
parent_parser.add_argument('-q', '--query_id', type=str, nargs='*')

formatting_group = parent_parser.add_argument_group('formatting')
formatting_group.add_argument('-fc', '--column', type=str, nargs='?', help='column to show')
formatting_group.add_argument('-fo', '--column_order', type=str, nargs='?', help='0-indexed column order')
formatting_group.add_argument('-fl', '--latex', action='store_true', help='display as latex table')
formatting_group.add_argument('-fp', '--precision', type=int, default=4, help='decimal precision')
formatting_group.add_argument('-fx', '--x_axis', type=str, default='', help='x axis name for chart')

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

bilingual_parser = subparsers.add_parser('bilingual', parents=[parent_parser])
bilingual_parser.add_argument('-de', '--doc_embed', type=str, help='Document-language embedding location')
bilingual_parser.add_argument('-qe', '--query_embed', type=str, help='Query-language embedding location')
bilingual_parser.set_defaults(func=bilingual(search_test))


args = parser.parse_args()

if args.collections == 'all':
    args.collections = list(readers.keys())

result = args.func([read_collection(base_dir=args.ir_dir, collection_name=name) for name in args.collections], args)

print_table(result, args)
# display_chart(result, args)
