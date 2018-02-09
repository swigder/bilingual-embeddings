import argparse
import os

from dictionary import MonolingualDictionary
from ir_data_reader import readers
from search_engine import SearchEngine


def precision_recall(expected, actual):
    true_positives = sum([1 for item in expected if item in actual])
    return true_positives / len(actual), true_positives / len(expected)


def read_data(data_dir, doc_file, query_file, relevance_file, reader):
    f = lambda file: os.path.join(data_dir, file)

    return reader.read(f(doc_file), f(query_file), f(relevance_file))


def test_search_engine(search_engine, documents, queries, relevance, n=5, verbose=False):
    search_engine.index_documents(documents.values())
    total_precision, total_recall = 0, 0
    for i, query in queries.items():
        results = search_engine.query_index(query, n_results=n)
        results_i = []
        if verbose:
            print()
            print(i, query, relevance[i])
        for distance, result in results:
            result_i = list(documents.keys())[list(documents.values()).index(result)]
            results_i.append(result_i)
            if verbose:
                print(result_i, distance, result[:300])
                if result_i in relevance[i]:
                    print('Correct!')
        precision, recall = precision_recall(expected=relevance[i], actual=results_i)
        total_precision += precision
        total_recall += recall
    print(total_precision / len(queries), total_recall / len(queries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('ir_dir', type=str, help='Directory with IR files', nargs='?')
    parser.add_argument('embed', type=str, help='Embedding file', nargs='?')
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')
    parser.add_argument('-n', '--number_results', type=int, default=5)
    parser.add_argument('-v', '--verbose', type=bool, default=False)

    args = parser.parse_args()

    if not args.ir_dir:
        args.ir_dir = '/Users/xx/Documents/school/kth/thesis/ir-datasets/'
    if not args.embed:
        args.embed = '/Users/xx/Downloads/MUSE-master/trained/vectors-en.txt'

    reader = readers[args.type](os.path.join(args.ir_dir, args.type))

    mono_dict = MonolingualDictionary(emb_file=args.embed)
    test_search_engine(SearchEngine(dictionary=mono_dict),
                       **reader.read_documents_queries_relevance(),
                       n=args.number_results,
                       verbose=args.verbose)
