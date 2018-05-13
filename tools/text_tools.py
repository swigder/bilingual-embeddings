"""
Normalize and tokenize the way fasttext does it - lowercase, convert digits to words, split on non-alpha.
"""
import argparse
from collections import defaultdict


def normalize(string):
    return string.lower()


def tokenize(string):
    return list(filter(bool,
                       string.replace('=', ' ')
                             .replace('-', ' - ')
                             .replace('/', ' / ')
                             .replace(',', ' , ')
                             .replace('.', ' . ')
                             .replace(';', ' ; ')
                             .replace(':', ' : ')
                             .replace('?', ' . ')
                             .replace('(', ' ( ')
                             .replace(')', ' ) ')
                             .replace('[', ' [ ')
                             .replace(']', ' ] ')
                             .replace("'", " ' ")
                             .replace('"', ' " ')
                             .replace('”', ' ” ')
                             .split())
                )


def f1_score(a, b):
    if a == 0 or b == 0:
        return 0
    return 2 * a * b / (a + b)


def detect_phrases(lines, cutoff=.1):
    unigrams = defaultdict(int)
    bigrams = defaultdict(int)
    for line in lines:
        words = line.tokenize()
        for word in words:
            unigrams[word] += 1
        for i in range(len(words)-1):
            bigrams['{}_{}'.format(words[i], words[i+1])] += 1
    bigram_probabilities = {}
    delta = sum(bigrams.values()) / (len(unigrams))
    for bigram, count in bigrams.items():
        if count < delta:
            continue
        word1, word2 = bigram.split('_')
        bigram_probabilities[bigram] = f1_score(count/unigrams[word1], count/unigrams[word2])
    return {bigram for bigram, probability in bigram_probabilities.items() if probability >= cutoff}


def replace_phrases(lines, phrases):
    phrases = {phrase: ' '.join(phrase.split('_')) for phrase in phrases}
    out_lines = []
    for line in lines:
        for phrase, original in phrases.items():
            line = line.replace(original, phrase)
        out_lines += line
    return out_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text tool.')

    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('output', type=str, help='output file')

    args = parser.parse_args()

    with open(args.input, 'r') as input:
        with open(args.output, 'w') as output:
            for line in input:
                output.write(' '.join(tokenize(normalize(line))) + '\n')


