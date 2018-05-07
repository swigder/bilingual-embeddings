import argparse
import operator
import os
import time

from collections import defaultdict

from text_tools import normalize, tokenize


start = time.time()


def print_with_time(string):
    elapsed = time.time() - start
    print('{} (+{:.2f}s) {}'.format(time.strftime('%a %H:%M:%S'), elapsed, string))


def process_article(article_text, dfs):
    tokens = set(tokenize(normalize(article_text)))
    for token in tokens:
        tokens.add(token)
        dfs[token] += 1


def compute_dfs(in_file_path, out_file_path, min_count=0, max_docs=-1, save_rate=100000, start_offset=0):
    if start_offset == 0:
        try:
            os.remove(out_file_path)  # so don't merge old data
        except FileNotFoundError:
            pass

    dfs = defaultdict(int)
    total_docs = 0
    offset = start_offset

    with open(in_file_path, 'r') as in_file:
        current_article = ''
        for line in in_file:
            if line[0] == '=' and line[1] != '=':  # new article
                total_docs += 1
                if 0 < start_offset < total_docs:
                    continue
                if total_docs % 10000 == 0:
                    print_with_time('Processing {} {}'.format(total_docs, line))
                if total_docs % save_rate == 0:
                    print_with_time('Saving at {} {}'.format(total_docs, line))
                    write_to_file(out_file_path, dfs, num_docs=total_docs-offset, merge=True)
                    dfs.clear()
                    offset = total_docs
                if -1 < max_docs <= total_docs:
                    break
                process_article(current_article, dfs)
                current_article = ''
            current_article += line.strip() + ' '
        process_article(current_article, dfs)

    write_to_file(out_file_path, dfs, num_docs=total_docs-offset, merge=True)

    dfs, num_docs = read_dfs(out_file_path)
    merge_low_df_files(out_file_path, dfs)
    write_to_file(out_file_path, dfs, num_docs=num_docs, min_count=min_count, merge=False)


def write_low_df_to_files(file_path, dfs):
    max_count = 5
    to_write = {i+1: [] for i in range(max_count)}
    for token, count in dfs.items():
        if count <= max_count:
            to_write[count].append(token)
    path, ext = os.path.splitext(file_path)
    for count, tokens in to_write.items():
        count_path = '{}-low-df-{}{}'.format(path, count, ext)
        with open(count_path, 'a') as file:
            for token in tokens:
                del dfs[token]
                file.write(token + '\n')


def merge_low_df_files(file_path, dfs):
    path, ext = os.path.splitext(file_path)
    for count in range(1, 6):
        count_path = '{}-low-df-{}{}'.format(path, count, ext)
        try:
            with open(count_path, 'r') as file:
                for line in file:
                    dfs[line.strip()] += count
            os.remove(count_path)
        except FileNotFoundError:
            pass


def delete_low_dfs(dfs, min_count):
    to_delete = set()
    for token, count in dfs.items():
        if count < min_count:
            to_delete.add(token)
    for token in to_delete:
        del dfs[token]


def write_to_file(file_path, dfs, num_docs, min_count=0, merge=False):
    if min_count == 0:
        write_low_df_to_files(file_path, dfs)

    if merge:
        try:
            dfs_2, num_docs_2 = read_dfs(file_path)
            num_docs += num_docs_2
            for k, v in dfs_2.items():
                dfs[k] += v
        except FileNotFoundError:
            pass  # nothing to merge

    if min_count > 0:
        delete_low_dfs(dfs, min_count)

    with open(file_path, 'w') as out_file:
        out_file.write('{} {}\n'.format(num_docs, len(dfs)))
        for token, count in sorted(dfs.items(), key=operator.itemgetter(1), reverse=True):
            out_file.write('{} {}\n'.format(token, count))


def read_dfs(file_name):
    dfs = defaultdict(int)
    with open(file_name, 'r') as file:
        num_docs, vocab_size = map(int, file.readline().strip().split())
        for line in file:
            token, count = line.strip().split()
            dfs[token] = int(count)
    assert vocab_size == len(dfs)
    return dfs, num_docs


def remove_low_dfs_from_file(input_file_name, output_file_name, min_count):
    dfs, num_docs = read_dfs(input_file_name)
    write_to_file(output_file_name, dfs, num_docs=num_docs, min_count=min_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Frequency Calculator')

    parser.add_argument('infile', type=str, help='Input file as parsed wiki')
    parser.add_argument('outfile', type=str, help='Output file')
    parser.add_argument('-m', '--min_count', type=int, default=0, help='Minimum count to output')
    parser.add_argument('-d', '--max_documents', type=int, default=-1, help='Maximum number of articles to process')
    parser.add_argument('-r', '--read', type=bool, default=False)

    args = parser.parse_args()

    compute_dfs(args.infile, args.outfile, min_count=args.min_count, max_docs=args.max_documents)
