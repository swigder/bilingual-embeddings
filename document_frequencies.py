import argparse
import operator
import sys
from collections import defaultdict

from nltk import word_tokenize


sub_tokens = {}


def process_article(article_text, dfs):
    rough_tokens = set(article_text.replace('=', ' ').replace(',', ' ').lower().split())
    tokens = set()
    for token in rough_tokens:
        if token.isalpha():
            tokens.add(token)
        elif token in sub_tokens:
            tokens.update(sub_tokens[token])
        else:
            tokenized = word_tokenize(token)
            sub_tokens[token] = tokenized
            tokens.update(tokenized)
    for token in tokens:
        dfs[token] += 1


def df(in_file, out_file, min_count):
    # Process input file
    dfs = defaultdict(int)
    total_docs = 0

    current_article = ''
    for line in in_file:
        if line[0] == '=' and line[1] != '=':  # new article
            total_docs += 1
            if total_docs % 1000 == 0:
                print(total_docs, line)
            process_article(current_article, dfs)
            current_article = ''
        current_article += line.strip() + ' '
    process_article(current_article, dfs)
    if in_file is not sys.stdout:
        in_file.close()

    # Remove anything below min
    if min_count > 0:
        to_delete = set()
        for token, count in dfs.items():
            if count < min_count:
                to_delete.add(token)
        for token in to_delete:
            del dfs[token]

    # Write to file
    out_file.write('{} {}\n'.format(total_docs, len(dfs)))
    for token, count in sorted(dfs.items(), key=operator.itemgetter(1), reverse=True):
        out_file.write('{} {}\n'.format(token, count))
    if out_file is not sys.stdout:
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Frequency Calculator')

    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w+'), default=sys.stdout)
    parser.add_argument('-m', '--min', type=int, default=0, help='Minimum count to output')

    args = parser.parse_args()

    df(args.infile, args.outfile, min_count=args.min)
