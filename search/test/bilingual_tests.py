import os

import pandas as pd

from dictionary import dictionary, BilingualDictionary, OovDictionary
from search_engine import BilingualEmbeddingSearchEngine
from utils import print_with_time


def bilingual(test):
    def inner(collections, parsed_args):
        if len(collections) != 1:
            raise ValueError
        collection = collections[0]

        embed_locations = parsed_args.embed_locations
        if parsed_args.search:
            from glob import glob
            embed_locations = []
            for embed_location in parsed_args.embed_locations:
                glob_format = '{}/**/{}'.format(embed_location, parsed_args.doc_embed)
                embed_locations += list(map(os.path.dirname, glob(glob_format, recursive=True)))
        print_with_time('Found {} locations to test.'.format(len(embed_locations)))

        df = pd.DataFrame(columns=test.columns, index=map(os.path.basename, embed_locations))

        for i, embed_location in enumerate(embed_locations):
            print_with_time('Testing ({}/{}) {}...'.format(i+1, len(embed_locations), embed_location))

            doc_dict = dictionary(os.path.join(embed_location, parsed_args.doc_embed), language='doc',
                                  use_subword=parsed_args.subword, normalize=parsed_args.normalize)
            query_dict = dictionary(os.path.join(embed_location, parsed_args.query_embed), language='query',
                                    use_subword=parsed_args.subword, normalize=parsed_args.normalize)
            if parsed_args.oov_embed:
                oov_dict = dictionary(os.path.join(embed_location, parsed_args.oov_embed), language='query',
                                      use_subword=parsed_args.subword, normalize=parsed_args.normalize)
                query_dict = OovDictionary([query_dict, doc_dict, oov_dict])
            bilingual_dictionary = BilingualDictionary(src_dict=doc_dict, tgt_dict=query_dict, default_lang='doc')

            bilingual_search_engine = BilingualEmbeddingSearchEngine(dictionary=bilingual_dictionary,
                                                                     doc_lang='doc', query_lang='query',
                                                                     query_df_file=parsed_args.df_file,
                                                                     use_weights=parsed_args.use_weights)
            bilingual_search_engine.index_documents(collection.documents.values())

            result = test.f(collection, bilingual_search_engine)
            df.loc[os.path.basename(embed_location)] = result
            print_with_time('Found MAP@10 {} for embed {}.'.format(result, embed_location))

        return df

    return inner
