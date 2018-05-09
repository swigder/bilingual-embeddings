import os

import pandas as pd

from dictionary import dictionary, BilingualDictionary
from search_engine import BilingualEmbeddingSearchEngine


def bilingual(test):
    def inner(collections, parsed_args):
        if len(collections) != 1:
            raise ValueError
        collection = collections[0]

        df = pd.DataFrame(columns=test.columns, index=map(os.path.basename, parsed_args.embed_locations))

        for embed_location in parsed_args.embed_locations:
            doc_dict = dictionary(os.path.join(embed_location, parsed_args.doc_embed), language='doc',
                                  use_subword=parsed_args.subword, normalize=parsed_args.normalize)
            query_dict = dictionary(os.path.join(embed_location, parsed_args.query_embed), language='query',
                                    use_subword=parsed_args.subword, normalize=parsed_args.normalize)
            bilingual_dictionary = BilingualDictionary(src_dict=doc_dict, tgt_dict=query_dict, default_lang='doc')

            bilingual_search_engine = BilingualEmbeddingSearchEngine(dictionary=bilingual_dictionary,
                                                                     doc_lang='doc', query_lang='query')
            bilingual_search_engine.index_documents(collection.documents.values())

            df.loc[os.path.basename(embed_location)] = test.f(collection, bilingual_search_engine)

        return df

    return inner
