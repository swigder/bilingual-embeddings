import argparse
import os

from collections import namedtuple, OrderedDict

from text_tools import tokenize, normalize, detect_phrases, replace_phrases

IrCollection = namedtuple('IrCollection', ['name', 'documents', 'queries', 'relevance'])
BilingualIrCollection = namedtuple('BilingualIrCollection', IrCollection._fields + ('queries_translated',))


def sub_collection(ir_collection, query):
    return IrCollection(name=ir_collection.name,
                        documents=ir_collection.documents,
                        queries={query: ir_collection.queries[query]},
                        relevance={query: ir_collection.relevance[query]})


def dir_appender(dir_location):
    return lambda file: os.path.join(dir_location, file)


class IrDataReader:
    def __init__(self, name, doc_file, query_file, relevance_file, translated_query_file=None):
        self.name = name
        self.doc_file = doc_file
        self.query_file = query_file
        self.relevance_file = relevance_file
        self.translated_query_file = translated_query_file

    def _read_file(self, file, extract_id_fn):
        items = OrderedDict()
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
        documents = self._read_file(self.doc_file, self.extract_doc_id)
        queries = self._read_file(self.query_file, self.extract_query_id)
        relevance_judgements = self.read_relevance_judgments()

        if not self.translated_query_file or not os.path.isfile(self.translated_query_file):
            return IrCollection(name=self.name,
                                documents=documents,
                                queries=queries,
                                relevance=relevance_judgements)

        queries_translated = self._read_file(self.translated_query_file, self.extract_query_id)

        return BilingualIrCollection(name=self.name,
                                     documents=documents,
                                     queries=queries,
                                     queries_translated=queries_translated,
                                     relevance=relevance_judgements)

    def read_relevance_judgments(self):
        items = OrderedDict()
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


class StandardReader(IrDataReader):
    def __init__(self, name, data_dir):
        f = lambda t: os.path.join(data_dir, '{}-{}.txt'.format(name, t))
        super().__init__(name=name,
                         doc_file=f('documents'),
                         query_file=f('queries'),
                         relevance_file=f('relevance'),
                         translated_query_file=f('queries_translated'))
        self.extract_doc_id = self.extract_id
        self.extract_query_id = self.extract_id

    @staticmethod
    def safe_int(string):
        try:
            return int(string)
        except ValueError:
            return string

    def extract_id(self, line):
        return None if not line.startswith('.id') else self.safe_int(line.split()[-1])

    def read_relevance_judgments(self):
        unsplit = self._read_file(self.relevance_file, self.extract_id)
        return {self.safe_int(rid): list(map(int, rels.split())) for rid, rels in unsplit.items()}

    def skip_line(self, line):
        return line.startswith('.count')


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
        # fix issue with offsets for certain relevance judgements due to missing documents
        doc_ids = [doc_id - 2 if doc_id >= 417 else doc_id - 1 if doc_id >= 413 else doc_id for doc_id in doc_ids]
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


readers = {'adi': AdiReader, 'time': TimeReader, 'ohsu-trec': OhsuReader}


def print_description(items, description):
    print()
    print('Read {} items in {}. Example items:'.format(len(items), description))
    keys = list(items.keys())
    print(keys[0], ':', items[keys[0]][:300])
    print(keys[1], ':', items[keys[1]][:300])


def read_collection(base_dir, collection_name, standard=True):
    path = os.path.join(base_dir, collection_name)
    reader = StandardReader(name=collection_name, data_dir=path) \
        if standard \
        else readers[collection_name](data_dir=path)
    return reader.read_documents_queries_relevance()


def describe_collection(collection, parsed_args):
    print('\nReading collection {}...'.format(collection.name))
    for name, item in collection._asdict().items():
        if name is 'name':
            continue
        print_description(item, name)


def write_fasttext_training_file(collection, parsed_args):
    def to_fasttext(line):
        return ' '.join(filter(lambda s: len(s) > 1, tokenize(normalize(line))))

    def write_lines_to_path(path, out_lines):
        with open(path, 'w') as out_file:
            out_file.write('\n'.join(out_lines))

    out_path = os.path.join(parsed_args.out_dir, collection.name + '.txt')

    if not parsed_args.phrases:
        with open(out_path, 'w') as file:
            for doc in collection.documents.values():
                file.write(to_fasttext(doc) + '\n')
    else:
        lines = [to_fasttext(doc) for doc in collection.documents.values()]
        phrases = detect_phrases(lines)
        write_lines_to_path('-phrases'.join(os.path.splitext(out_path)), phrases)
        write_lines_to_path(out_path, replace_phrases(lines, phrases))


def write_to_standard_form(collection, parsed_args):
    def write_to_file(name, dictionary, transform=lambda i: i):
        out_path = os.path.join(parsed_args.out_dir, collection.name, '{}-{}.txt'.format(collection.name, name))
        with open(out_path, 'w') as out_file:
            out_file.write('.count.{} {}\n\n'.format(name, len(dictionary)))
            for item_id, item in dictionary.items():
                out_file.write('.id.{} {}\n{}\n\n'.format(name[0], item_id, transform(item)))

    os.makedirs(os.path.join(parsed_args.out_dir, collection.name), exist_ok=True)

    if parsed_args.phrases:
        phrases = detect_phrases([normalize(doc) for doc in collection.documents])
        transform_f = lambda l: ' '.join(tokenize(replace_phrases([normalize(l)], phrases=phrases)))
    else:
        transform_f = lambda l: ' '.join(tokenize(normalize(l)))

    write_to_file('documents', collection.documents, transform=transform_f)
    write_to_file('queries', collection.queries, transform=transform_f)
    write_to_file('relevance', collection.relevance, transform=lambda l: ' '.join(map(str, l)))


if __name__ == "__main__":
    def split_calls(f):
        return lambda cs, a: [f(c, a) for c in cs]

    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('dir', type=str, help='Directory with files')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-c', '--collections', nargs='+', choices=list(readers.keys()) + ['all'], default='all')

    subparsers = parser.add_subparsers()

    parser_describe = subparsers.add_parser('describe', parents=[parent_parser])
    parser_describe.add_argument('-s', '--standard', action='store_true')
    parser_describe.set_defaults(func=split_calls(describe_collection))

    parser_fasttext = subparsers.add_parser('fasttext', parents=[parent_parser])
    parser_fasttext.add_argument('out_dir', type=str, help='Output directory')
    parser_fasttext.add_argument('-p', '--phrases', action='store_true', help='Detect and join phrases.')
    parser_fasttext.add_argument('-s', '--standard', action='store_true')
    parser_fasttext.set_defaults(func=split_calls(write_fasttext_training_file))

    parser_fasttext = subparsers.add_parser('standardize', parents=[parent_parser])
    parser_fasttext.add_argument('-p', '--phrases', action='store_true', help='Detect and join phrases.')
    parser_fasttext.add_argument('out_dir', type=str, help='Output directory')
    parser_fasttext.set_defaults(func=split_calls(write_to_standard_form))
    parser_fasttext.set_defaults(standard=False)

    args = parser.parse_args()

    if args.collections == 'all':
        args.collections = list(readers.keys())

    collections = [read_collection(base_dir=args.dir, collection_name=name, standard=args.standard)
                   for name in args.collections]
    args.func(collections, args)
