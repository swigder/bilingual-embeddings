import argparse
import os

from collections import namedtuple

from text_tools import tokenize, normalize


IrCollection = namedtuple('IrCollection', ['name', 'documents', 'queries', 'relevance'])


def sub_collection(ir_collection, query):
    return IrCollection(name=ir_collection.name,
                        documents=ir_collection.documents,
                        queries={query: ir_collection.queries[query]},
                        relevance={query: ir_collection.relevance[query]})


def dir_appender(dir_location):
    return lambda file: os.path.join(dir_location, file)


class IrDataReader:
    def __init__(self, name, doc_file, query_file, relevance_file):
        self.name = name
        self.doc_file = doc_file
        self.query_file = query_file
        self.relevance_file = relevance_file

    def _read_file(self, file, extract_id_fn):
        items = {}
        current_id = ''
        current_item = ''

        with open(file) as f:
            for line in f:
                line = line.strip()
                if not line or self.skip_line(line):
                    continue
                doc_id = extract_id_fn(line)
                if doc_id is None:
                    current_item += self.extract_line(line.strip()) + ' '
                else:
                    if current_item:
                        items[current_id] = current_item.lower()
                    current_id = doc_id
                    current_item = ''
            items[current_id] = current_item.lower()

        return items

    def read_documents_queries_relevance(self):
        return IrCollection(name=self.name,
                            documents=self.read_documents(),
                            queries=self.read_queries(),
                            relevance=self.read_relevance_judgments())

    def read_documents(self):
        return self._read_file(self.doc_file, self.extract_doc_id)

    def read_queries(self):
        return self._read_file(self.query_file, self.extract_query_id)

    def read_relevance_judgments(self):
        items = {}
        with open(self.relevance_file) as f:
            for line in f:
                if not line.strip():
                    continue
                query_id, doc_ids = self.extract_relevance(line)
                if query_id not in items:
                    items[query_id] = []
                items[query_id] += doc_ids
        return items

    def extract_doc_id(self, line):
        pass

    def extract_query_id(self, line):
        pass

    def extract_line(self, line):
        return line

    def skip_line(self, line):
        return False

    def extract_relevance(self, line):
        pass


class TimeReader(IrDataReader):
    def __init__(self, data_dir):
        f = dir_appender(data_dir)
        super().__init__(name='time', doc_file=f('TIME.ALL'), query_file=f('TIME.QUE'), relevance_file=f('TIME.REL'))
        self.id = 0

    def extract_doc_id(self, line):
        if not line.startswith('*TEXT'):
            return None
        self.id += 1
        return self.id

    def extract_query_id(self, line):
        if not line.startswith('*FIND'):
            return None
        return int(line.split()[1].strip())

    def extract_relevance(self, line):
        query_id, *doc_ids = map(int, line.split())
        return query_id, doc_ids

    def skip_line(self, line):
        return line.startswith('*STOP')


class AdiReader(IrDataReader):
    def __init__(self, data_dir):
        f = dir_appender(data_dir)
        super().__init__(name='adi', doc_file=f('ADI.ALL'), query_file=f('ADI.QRY'), relevance_file=f('ADI.REL'))

    @staticmethod
    def extract_id(line):
        if not line.startswith('.I'):
            return None
        return int(line.split()[1])

    def extract_doc_id(self, line):
        return self.extract_id(line)

    def extract_query_id(self, line):
        return self.extract_id(line)

    def extract_relevance(self, line):
        query_id, doc_id = map(int, line.split()[0:2])
        return query_id, [doc_id]

    def skip_line(self, line):
        return line.startswith('.') and not line.startswith('.I')


class OhsuReader(IrDataReader):
    def __init__(self, data_dir):
        f = dir_appender(os.path.join(data_dir, 'trec9-train'))
        super().__init__(name='ohsu-trec',
                         doc_file=f('ohsumed.87'),
                         query_file=f('query.ohsu.1-63'),
                         relevance_file=f('qrels.ohsu.batch.87'))
        self.previous_line_marker = None

    def extract_doc_id(self, line):
        if self.previous_line_marker == '.U':
            self.previous_line_marker = None
            return int(line)
        return None

    def extract_query_id(self, line):
        if line.startswith('<num>'):
            return line.split(':')[1].strip()

    def extract_relevance(self, line):
        query_id, doc_id = line.split()[0:2]
        return query_id, [int(doc_id)]

    def extract_line(self, line):
        if line.startswith('<title>'):
            return line[8:]
        return line

    def skip_line(self, line):
        if line.startswith('.'):
            self.previous_line_marker = line
            return True
        if line.startswith('<desc>') or line.startswith('<top>') or line.startswith('</top>'):
            return True
        if self.previous_line_marker in ['.S', '.M', '.P', '.A']:  # skip all fields except uid, title, abstract
            self.previous_line_marker = None
            return True
        return False


readers = {'time': TimeReader, 'adi': AdiReader, 'ohsu-trec': OhsuReader}


def print_description(items, description):
    print()
    print('Read {} items in {}. Example items:'.format(len(items), description))
    keys = list(items.keys())
    print(keys[0], ':', items[keys[0]][:300])
    print(keys[1], ':', items[keys[1]][:300])


def read_collection(base_dir, collection_name):
    reader = readers[collection_name](os.path.join(base_dir, collection_name))
    return reader.read_documents_queries_relevance()


def describe_collection(collection, parsed_args):
    print('\nReading collection {}...'.format(collection.name))
    for name, item in collection._asdict().items():
        if name is 'name':
            continue
        print_description(item, name)


def write_fasttext_training_file(collection, parsed_args):
    out_path = os.path.join(parsed_args.out_dir, collection.name + '-fasttext-training.txt')

    with open(out_path, 'w') as file:
        for doc in collection.documents.values():
            file.write(' '.join(filter(lambda s: len(s) > 1, tokenize(normalize(doc)))) + '\n')


if __name__ == "__main__":
    def split_calls(f):
        return lambda cs, a: [f(c, a) for c in cs]

    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('dir', type=str, help='Directory with files')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-t', '--types', nargs='+', choices=list(readers.keys()) + ['all'], default='all')

    subparsers = parser.add_subparsers()

    parser_describe = subparsers.add_parser('describe', parents=[parent_parser])
    parser_describe.set_defaults(func=split_calls(describe_collection))

    parser_fasttext = subparsers.add_parser('fasttext', parents=[parent_parser])
    parser_fasttext.add_argument('out_dir', type=str, help='Output directory')
    parser_fasttext.set_defaults(func=split_calls(write_fasttext_training_file))

    args = parser.parse_args()

    if args.types == 'all':
        args.types = list(readers.keys())

    args.func([read_collection(base_dir=args.dir, collection_name=name) for name in args.types], args)
