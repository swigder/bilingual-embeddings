"""
Normalize and tokenize the way fasttext does it - lowercase, split on non-alpha.
"""


def normalize(string):
    return string.lower()


def tokenize(string):
    return list(filter(bool,
                       string.replace('=', ' ')
                             .replace('-', ' - ')
                             .replace(',', ' , ')
                             .replace('.', ' . ')
                             .replace('(', ' ) ')
                             .replace(')', ' ( ')
                             .replace("'", " ' ")
                             .replace('"', ' " ')
                             .split()))
