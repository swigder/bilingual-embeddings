import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_and_combine(path):
    df = pd.read_pickle(path)
    df = df.astype(float).groupby(df.index).mean()
    try:
        cols = list(map(str, sorted(map(int, df.columns.tolist()))))
        df = df[cols]
    except ValueError:
        pass
    return df


def add_percentage_change(df, from_col, to_col):
    df['Change'] = (df[to_col] - df[from_col]) / df[from_col] * 100


def load_change_show(path, from_col, to_col):
    df = load_and_combine(path)
    add_percentage_change(df, from_col, to_col)
    pd.set_option('precision', 4)
    print(df.to_latex())
    return df


def map_files_to_df_components(file_string):
    mapping = {}
    for line in file_string.split('\n'):
        if not line:
            continue
        file = line.split()[-1]
        parts = file.split('/')
        mapping[parts[-2]] = parts[-4]

    df = pd.DataFrame(index=mapping.keys(), columns=['collection', 'sub', 'win', 'epoch', 'min'])
    for k, v in mapping.items():
        split = v.split('-')
        d = {'collection': '-'.join(split[:3])}
        for i, name in enumerate(split):
            if name in df.columns:
                d[name] = split[i + 1]
        df.loc[k] = d

    return df


def file_name_only(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_all_pickles_in_dir(path):
    assert os.path.isdir(path)
    return pd.concat([read_grid_pickle(file) for file in glob.glob(os.path.join(path, '*.pkl'))], axis=0)


def read_grid_pickle(path):
    results = pd.read_pickle(path)
    results.index = results.index.droplevel()

    files_name = file_name_only(path)
    _, collection, norm, subword = files_name.split('_')[0].split('-')
    assert norm == 'nn' or norm == 'norm'
    assert subword == 'subword' or subword == 'zero'
    hyperparams = pd.DataFrame(index=results.index,
                               columns=['collection', 'sub', 'win', 'epochs', 'norm', 'subword'])
    for result in hyperparams.index:
        parts = file_name_only(result).split('-')
        assert parts[0] == collection
        d = {'collection': collection, 'norm': norm, 'subword': subword}
        for i, name in enumerate(parts):
            if name in hyperparams.columns:
                d[name] = parts[i + 1]
        hyperparams.loc[result] = d

    return pd.concat([results, hyperparams], axis=1).apply(pd.to_numeric, errors='ignore')


def unpickle_multiple(prefix, files):
    return pd.concat([pd.read_pickle(os.path.join(prefix, file)) for file in files])


def get_results(file_string, prefix, files):
    r = pd.concat([map_files_to_df_components(file_string), unpickle_multiple(prefix, files)], axis=1)
    return r.apply(pd.to_numeric, errors='ignore')


def get_max_map(df):
    return df.loc[df['MAP@10'].idxmax()]


def two_2_map(df, row, col):
    return df.groupby([row, col])['MAP@10'].mean().unstack()


def analyze(df):
    sns.set()
    attributes_to_test = ['sub', 'win', 'epochs', 'norm', 'subword']
    collection_groups = df.groupby('collection')
    for collection in collection_groups.groups:
        collection_df = collection_groups.get_group(collection)
        for attribute in attributes_to_test:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            others = [a for a in attributes_to_test if a != attribute]
            for i, other in enumerate(others):
                two_2_map(collection_df, attribute, other).plot(ax=axes[i//2, i%2])
            plt.suptitle('{} {}'.format(collection, attribute))
            plt.subplots_adjust(top=.9)
            plt.show()
