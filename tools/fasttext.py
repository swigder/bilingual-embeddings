import argparse
import datetime

import fastText


def shrink_model(model_file, vocab_file, out_file, fast):
    print(datetime.datetime.now(), 'Reading vocabulary...')
    words = []
    with open(vocab_file) as f:
        for line in f:
            words.append(line.strip())

    vectors = {}
    print(datetime.datetime.now(), 'Reading model...')
    model = fastText.load_model(path=model_file)
    print(datetime.datetime.now(), 'Shrinking model...')
    if fast:
        for word in words:
            vectors[word] = model.get_word_vector(word)
    else:
        for word in model.get_words():
            if word.lower() in words:
                vectors[word] = model.get_word_vector(word)

    print(datetime.datetime.now(), 'Writing small model...')
    with open(out_file, 'w') as f:
        f.write('{} {}\n'.format(len(vectors), model.get_dimension()))
        for word, vector in vectors.items():
            f.write('{} {}\n'.format(word, ' '.join(map(str, vector))))

    print(datetime.datetime.now(), 'Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fasttext tool.')

    parser.add_argument('model', type=str, help='binary model file')
    parser.add_argument('vocab', type=str, help='text vocabulary file')
    parser.add_argument('out', type=str, help='output file')
    parser.add_argument('-f', '--fast', action='store_true')

    args = parser.parse_args()

    shrink_model(model_file=args.model, vocab_file=args.vocab, out_file=args.out, fast=args.fast)
