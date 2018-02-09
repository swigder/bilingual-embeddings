import argparse


class IrDataReader:
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

    def _read_relevance(self, file):
        items = {}
        with open(file) as f:
            for line in f:
                if not line.strip():
                    continue
                query_id, doc_ids = self.extract_relevance(line)
                if query_id not in items:
                    items[query_id] = []
                items[query_id] += doc_ids
        return items

    def read(self, documents=None, queries=None, relevance=None):
        ret = []
        if documents:
            ret.append(self.read_documents(documents))
        if queries:
            ret.append(self.read_queries(queries))
        if relevance:
            ret.append(self.read_relevance_judgments(relevance))
        return ret

    def read_documents(self, file):
        return self._read_file(file, self.extract_doc_id)

    def read_queries(self, file):
        return self._read_file(file, self.extract_query_id)

    def read_relevance_judgments(self, file):
        return self._read_relevance(file)

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
    def __init__(self):
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
    @staticmethod
    def extract_id(line):
        if not line.startswith('.I'):
            return None
        return line.split()[1]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('dir', type=str, help='Directory with files')
    parser.add_argument('-d', '--documents', type=str, help='Documents file', required=False)
    parser.add_argument('-q', '--queries', type=str, help='Queries file', required=False)
    parser.add_argument('-r', '--relevance', type=str, help='Relevance file', required=False)
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')

    args = parser.parse_args()

    reader = readers[args.type]

    def print_description(items, file):
        print()
        print('Read {} items in file {}. Example items:'.format(len(items), file))
        keys = list(items.keys())
        print('{}: {}'.format(keys[0], items[keys[0]][:300]))
        print('{}: {}'.format(keys[1], items[keys[1]][:300]))

    if args.documents:
        print_description(reader().read_documents(args.dir + args.documents), args.documents)
    if args.queries:
        print_description(reader().read_queries(args.dir + args.queries), args.queries)
    if args.relevance:
        print_description(reader().read_relevance_judgments(args.dir + args.relevance), args.relevance)
