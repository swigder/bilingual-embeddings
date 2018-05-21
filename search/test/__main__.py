import argparse

from .df_tests import vary_df, add_df_parser_options
from .oov_tests import oov_test
from .testing_framework import vary_embeddings, search_test, embed_to_engine, hyperparameters, multirun_map, recall_test
from .bilingual_tests import bilingual
from .testing_io import print_table, display_chart, save_to_file
from ir_data_reader import readers, read_collection


parser = argparse.ArgumentParser(description='Run tests on search engine.')
subparsers = parser.add_subparsers(title='subcommands')

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('ir_dir', type=str, help='Directory with IR files')
parent_parser.add_argument('-c', '--collections', choices=list(readers.keys()) + ['all'], nargs='*', default='all')
parent_parser.add_argument('-u', '--untranslated', action='store_true')
parent_parser.add_argument('-b', '--baseline', action='store_true')
parent_parser.add_argument('-s', '--subword', action='store_true')
parent_parser.add_argument('-n', '--normalize', action='store_true')
parent_parser.add_argument('-a', '--all_dictionary_options', action='store_true')
parent_parser.add_argument('-q', '--query_id', type=str, nargs='*')
parent_parser.add_argument('-m', '--multirun', action='store_true')
parent_parser.add_argument('--save_file', type=str, default='')

formatting_group = parent_parser.add_argument_group('formatting')
formatting_group.add_argument('-fo', '--column_order', type=str, nargs='?', help='0-indexed column order')
formatting_group.add_argument('-fl', '--latex', action='store_true', help='display as latex table')
formatting_group.add_argument('-fp', '--precision', type=int, default=4, help='decimal precision')
formatting_group.add_argument('-fx', '--x_axis', type=str, default='', help='x axis name for chart')
formatting_group.add_argument('-fy', '--y_axis', type=str, default='', help='y axis name for chart')
formatting_group.add_argument('-ft', '--title', type=str, default='', help='title for chart')

monolingual_parent_parser = argparse.ArgumentParser(add_help=False, parents=[parent_parser])
embedding_group = monolingual_parent_parser.add_argument_group('embedding')
embedding_group.add_argument('-E', '--embed_location', type=str, nargs='?',
                             help='Embedding directory; can be used instead of full path in other domain args')
embedding_group.add_argument('-D', '--domain_embed', type=str, nargs='*',
                             help='Embedding format for domain-specific embedding')
embedding_group.add_argument('-Q', '--embed', type=str, nargs='*',
                             help='Embedding location for general purpose embedding')

interactive_parser = subparsers.add_parser('interactive', parents=[monolingual_parent_parser])
interactive_parser.set_defaults(func=None)

oov_parser = subparsers.add_parser('oov', parents=[monolingual_parent_parser])
oov_parser.set_defaults(func=vary_embeddings(oov_test))

embedding_search_parser = subparsers.add_parser('embed', parents=[monolingual_parent_parser])
embedding_search_parser.set_defaults(func=vary_embeddings(embed_to_engine(search_test)))

recall_search_parser = subparsers.add_parser('recall', parents=[monolingual_parent_parser])
recall_search_parser.set_defaults(func=vary_embeddings(embed_to_engine(recall_test)))

hyperparameters_search_parser = subparsers.add_parser('hyperparameters', parents=[monolingual_parent_parser])
hyperparameters_search_parser.add_argument('-r', '--relative', type=str, nargs='?', help='column to show relative to')
hyperparameters_search_parser.set_defaults(func=hyperparameters(embed_to_engine(search_test)))

df_parser = subparsers.add_parser('df', parents=[monolingual_parent_parser])
add_df_parser_options(df_parser)
df_parser.set_defaults(func=vary_df(search_test))

bilingual_parser = subparsers.add_parser('bilingual', parents=[parent_parser])
bilingual_parser.add_argument('-E', '--embed_locations', type=str, nargs='+', help='Embedding directory')
bilingual_parser.add_argument('-D', '--doc_embed', type=str, help='Document-language embedding file')
bilingual_parser.add_argument('-Q', '--query_embed', type=str, help='Query-language embedding file')
bilingual_parser.add_argument('-df', '--df_file', type=str, help='File for query-language df')
bilingual_parser.add_argument('-uw', '--use_weights', action='store_true', help='Use df weights (not just stopwords)')
bilingual_parser.add_argument('--search', action='store_true', help='Search for subdirectories containing embeddings')
bilingual_parser.set_defaults(func=bilingual(search_test))


args = parser.parse_args()

if args.collections == 'all':
    args.collections = list(readers.keys())

collections = [read_collection(base_dir=args.ir_dir, collection_name=name, untranslated=args.untranslated) for name in args.collections]

if args.func:
    result = multirun_map(args.func)(collections, args)

    if not args.x_axis:
        print_table(result, args)
    else:
        display_chart(result, args)

    save_to_file(result, args)