"""
Normalize and tokenize the way fasttext does it - lowercase, convert digits to words, split on non-alpha.
"""


def normalize(string):
    return (string.lower()
                  .replace('1', ' one ')
                  .replace('2', ' two ')
                  .replace('3', ' three ')
                  .replace('4', ' four ')
                  .replace('5', ' five ')
                  .replace('6', ' six ')
                  .replace('7', ' seven ')
                  .replace('8', ' eight ')
                  .replace('9', ' nine ')
    )


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
