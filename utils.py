import time
from collections import defaultdict


start = None


def print_with_time(string, restart=False):
    global start
    if not start or restart:
        start = time.time()
    elapsed = time.time() - start
    print('{} (+{:.2f}s) {}'.format(time.strftime('%a %H:%M:%S'), elapsed, string))


def read_dfs(file_name):
    dfs = defaultdict(int)
    with open(file_name, 'r') as file:
        num_docs, vocab_size = map(int, file.readline().strip().split())
        for line in file:
            token, count = line.strip().split()
            dfs[token] = int(count)
    assert vocab_size == len(dfs)
    return dfs, num_docs
