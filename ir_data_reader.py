import argparse
import os

from collections import namedtuple


IrCollection = namedtuple('IrCollection', ['documents', 'queries', 'relevance'])


def sub_collection(ir_collection, query):
    return IrCollection(ir_collection.documents,
                        {query: ir_collection.queries[query]},
                        {query: ir_collection.relevance[query]})


def dir_appender(dir_location):
    return lambda file: os.path.join(dir_location, file)


class IrDataReader:
    def __init__(self, doc_file, query_file, relevance_file):
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
        return IrCollection(documents=self.read_documents(),
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
        super().__init__(doc_file=f('TIME.ALL'), query_file=f('TIME.QUE'), relevance_file=f('TIME.REL'))
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
        super().__init__(doc_file=f('ADI.ALL'), query_file=f('ADI.QRY'), relevance_file=f('ADI.REL'))

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
        super().__init__(doc_file=f('ohsumed.87'),
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


def print_query_oov_rate(ir_collection):
    from text_tools import tokenize, normalize
    document_tokens = set()
    for document in ir_collection.documents.values():
        document_tokens.update(tokenize(normalize(document)))
    in_vocabulary = 0
    out_of_vocabulary = 0
    for query in ir_collection.queries.values():
        for token in tokenize(normalize(query)):
            if token in document_tokens:
                in_vocabulary += 1
            else:
                out_of_vocabulary += 1
    print()
    print('In vocabulary {}, Out of vocabulary {}, OOV rate {}'
          .format(in_vocabulary,
                  out_of_vocabulary,
                  out_of_vocabulary / (in_vocabulary + out_of_vocabulary)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('dir', type=str, help='Directory with files')
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')

    args = parser.parse_args()

    reader = readers[args.type](os.path.join(args.dir, args.type))

    ir_collection = reader.read_documents_queries_relevance()
    for name, item in ir_collection._asdict().items():
        print_description(item, name)

    print_query_oov_rate(ir_collection)
