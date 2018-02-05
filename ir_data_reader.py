import argparse


class IrDataReader:
    def _read_file(self, file, extract_id_fn):
        items = {}
        current_id = ''
        current_item = ''

        with open(file) as f:
            for line in f:
                if not line:
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

    @staticmethod
    def _read_relevance(file):
        items = {}
        with open(file) as f:
            for line in f:
                if not line.strip():
                    continue
                query_id, *doc_ids = map(int, line.split())
                items[query_id] = doc_ids
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

    def extract_line(self, line):
        return line


readers = {'time': TimeReader}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('dir', type=str, help='Directory with files')
    parser.add_argument('-d', '--documents', type=str, help='Documents file', required=False)
    parser.add_argument('-q', '--queries', type=str, help='Queries file', required=False)
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')

    args = parser.parse_args()

    if args.documents:
        TimeReader().read_documents(args.dir + args.documents)
    if args.queries:
        TimeReader().read_queries(args.dir + args.queries)
