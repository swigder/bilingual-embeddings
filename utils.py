from collections import defaultdict


def read_dfs(file_name):
    dfs = defaultdict(int)
    with open(file_name, 'r') as file:
        num_docs, vocab_size = map(int, file.readline().strip().split())
        for line in file:
            token, count = line.strip().split()
            dfs[token] = int(count)
    assert vocab_size == len(dfs)
    return dfs, num_docs
