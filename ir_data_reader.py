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
        doc_id, *judgements = map(int, line.split())
        return doc_id, judgements


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
        doc_id, judgements = map(int, line.split()[0:2])
        return doc_id, [judgements]

    def skip_line(self, line):
        return line.startswith('.') and not line.startswith('.I')


readers = {'time': TimeReader, 'adi': AdiReader}


def print_description(items, description):
    print()
    print('Read {} items in {}. Example items:'.format(len(items), description))
    keys = list(items.keys())
    print(keys[0], ':', items[keys[0]][:300])
    print(keys[1], ':', items[keys[1]][:300])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('dir', type=str, help='Directory with files')
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')

    args = parser.parse_args()

    reader = readers[args.type](os.path.join(args.dir, args.type))

    for name, item in reader.read_documents_queries_relevance()._asdict().items():
        print_description(item, name)
