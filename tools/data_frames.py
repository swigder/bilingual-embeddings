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


def read_all_pickles_in_dirs(paths):
    return pd.concat([read_all_pickles_in_dir(path) for path in paths], axis=0).drop_duplicates()


def read_all_pickles_in_dir(path):
    assert os.path.isdir(path)
    return pd.concat([read_grid_pickle(file) for file in glob.glob(os.path.join(path, '*.pkl'))], axis=0).drop_duplicates()


def read_grid_pickle(path):
    results = pd.read_pickle(path)

    files_name = file_name_only(path)
    _, collection, pretrained, norm, subword = files_name.split('_')[0].split('-')

    assert pretrained == 'only' or pretrained == 'wiki'
    pretrained = pretrained == 'wiki'

    contains_all = norm == 'all' and subword == 'all'
    if not contains_all:
        results.index = results.index.droplevel()
        assert norm == 'nn' or norm == 'norm'
        assert subword == 'subword' or subword == 'zero'
        norm = norm == 'norm'
        subword = subword == 'subword'
    columns = ['sub', 'win', 'epochs', 'pretrained']
    if not contains_all:
        columns += ['collection', 'norm', 'subword']
    hyperparams = pd.DataFrame(index=results.index, columns=columns)
    for result in hyperparams.index:
        file_name = result if not contains_all else results.at[result, 'embedding']
        parts = file_name_only(file_name).split('-')
        assert parts[0] == collection
        d = {'pretrained': pretrained}
        d.update({'collection': collection, 'norm': norm, 'subword': subword} if not contains_all else {})
        for i, name in enumerate(parts):
            if name in hyperparams.columns:
                d[name] = parts[i + 1]
        hyperparams.loc[result] = d

    results = pd.concat([results, hyperparams], axis=1)
    if contains_all:
        results = results.set_index('embedding')
    results['collection'] = results['collection'].replace(['ohsu-trec'], 'ohsu')
    results['sub'] = results['sub'].replace(['7'], 'No')
    return results.apply(pd.to_numeric, errors='ignore')


def unpickle_multiple(prefix, files):
    return pd.concat([pd.read_pickle(os.path.join(prefix, file)) for file in files]).drop_duplicates()


def get_results(file_string, prefix, files):
    r = pd.concat([map_files_to_df_components(file_string), unpickle_multiple(prefix, files)], axis=1)
    return r.apply(pd.to_numeric, errors='ignore')


def get_max_map(df):
    return df.loc[df['MAP@10'].idxmax()]


def two_2_map(df, row, col):
    return df.groupby([row, col])['MAP@10'].mean().unstack()


def analyze(df):
    def a(sub_df, name, attributes_to_test, attributes_to_control, reverse=False):
        for attribute in attributes_to_test:
            fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)
            others = [a for a in attributes_to_control if a != attribute]
            for i, other in enumerate(others):
                a, b = (attribute, other) if not reverse else (other, attribute)
                grid = two_2_map(sub_df, a, b)
                grid.plot(ax=axes[i // 2, i % 2])
            values = grid.index.tolist()
            if not reverse:
                plt.setp(axes, xticks=values if all(type(x) is int for x in values) else range(len(values)),
                         xticklabels=values)
            plt.suptitle('{} {}'.format(name, attribute))
            plt.show()

    sns.set()
    attributes = ['sub', 'win', 'epochs', 'norm', 'subword']
    a(df, 'all', ['collection'], attributes[:4], reverse=True)
    a(df, 'all', attributes, attributes)
    collection_groups = df.groupby('collection')
    for collection in collection_groups.groups:
        collection_df = collection_groups.get_group(collection)
        a(collection_df, collection, attributes, attributes)


def overall_parameters(df, baselines=False):
    attributes = ['sub', 'win', 'epochs', 'norm', 'subword', 'pretrained']
    baseline_columns = {'sub': 'No', 'win': 5, 'epochs': 5, 'norm': False, 'subword': False, 'pretrained': False}
    sns.set()
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True)
    for i, attribute in enumerate(attributes):
        a = df.groupby([attribute, 'collection']).mean()['MAP@10'].unstack()
        if baselines:
            baseline_values = df.where(df[attribute] == baseline_columns[attribute]).groupby('collection').mean()['MAP@10']
            a = (a - baseline_values) / baseline_values
        current_axes = axes[i // 3, i % 3]
        a.plot(ax=current_axes)
        values = a.index.tolist()
        current_axes.set_xticks(values if all(type(x) is int for x in values) else range(len(values)))
        current_axes.set_xticklabels(values)
    plt.show()


def impact_of_parameters(df):
    attributes = ['sub', 'win', 'epochs', 'norm', 'subword', 'pretrained']
    stds = pd.DataFrame(index=df.groupby('collection').groups, columns=attributes)
    for attribute in attributes:
        those = df.groupby(['collection', attribute]).mean()['MAP@10'].unstack()
        stds[attribute] = (those.max(axis=1) - those.min(axis=1)) / those.min(axis=1)
    return stds

