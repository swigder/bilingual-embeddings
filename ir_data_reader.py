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
                if doc_id is not None:
                    items[current_id] = current_item.lower()
                    current_id = doc_id
                    current_item = ''
                else:
                    current_item += self.extract_line(line.strip()) + ' '
            items[current_id] = current_item

        return items

    def read_documents(self, file):
        return self._read_file(file, self.extract_doc_id)

    def read_queries(self, file):
        return self._read_file(file, self.extract_query_id)

    def extract_doc_id(self, line):
        pass

    def extract_query_id(self, line):
        pass

    def extract_line(self, line):
        pass


class TimeReader(IrDataReader):
    def extract_doc_id(self, line):
        if not line.startswith('*TEXT'):
            return None
        return line.split(' ')[1]

    def extract_query_id(self, line):
        if not line.startswith('*FIND'):
            return None
        return line.split(' ' * 4)[1].strip()

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
