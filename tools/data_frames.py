import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import gridspec


attributes = ['sub', 'win', 'epochs', 'norm', 'subword', 'pretrained']
nice_names = {'sub': 'Minimum subword length',
              'win': 'Window size',
              'epochs': 'Epochs',
              'norm': 'Normalize length',
              'subword': 'Use subword data for OOV',
              'pretrained': 'Model type'}
baseline_columns = {'sub': 'No',
                    'win': 5,
                    'epochs': 5,
                    'norm': False,
                    'subword': False,
                    'pretrained': 'Collection'}
replacements = {'collection': {'ohsu-trec': 'ohsu'},
                'sub': {'7': 'No'},
                'pretrained': {True: 'Hybrid', False: 'Collection'}}


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
    df = pd.concat([read_grid_pickle(file) for file in glob.glob(os.path.join(path, '*.pkl'))], axis=0)
    return df.drop_duplicates(df.columns.difference(['MAP@10']))


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

    for attribute, pairs in replacements.items():
        for original, replaced in pairs.items():
            results[attribute] = results[attribute].replace([original], replaced)
    return results.apply(pd.to_numeric, errors='ignore')


def unpickle_multiple(prefix, files):
    return pd.concat([pd.read_pickle(os.path.join(prefix, file)) for file in files]).drop_duplicates()


def get_results(file_string, prefix, files):
    r = pd.concat([map_files_to_df_components(file_string), unpickle_multiple(prefix, files)], axis=1)
    return r.apply(pd.to_numeric, errors='ignore')


def get_max_map(df):
    return df.loc[df.groupby('collection')['MAP@10'].idxmax()]


def two_2_map(df, row, col):
    return df.groupby([row, col])['MAP@10'].mean().unstack()


def parameter_interaction(df):
    sns.set()

    for attribute in attributes:
        others = [a for a in attributes if a != attribute]

        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(nice_names[attribute])
        outer = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.1)

        collection_groups = df.groupby('collection')
        for i, collection in enumerate(collection_groups.groups):
            collection_df = collection_groups.get_group(collection)

            inner = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            axes = np.empty(shape=(1, 5), dtype=object)
            for j in range(len(others)):
                ax = plt.Subplot(fig, inner[j], sharex=axes[0, 0], sharey=axes[0, 0])
                fig.add_subplot(ax)
                axes[0, j] = ax

            for j, other in enumerate(others):
                grid = two_2_map(collection_df, attribute, other)
                ax = axes[0, j]
                grid.plot(ax=ax)
                values = grid.index.tolist()
                if ax.is_last_row():
                    ax.set_xticks(values if all(type(x) is int for x in values) else range(len(values)))
                    ax.set_xticklabels(values)
                    if i != 2 or j != 2:
                        ax.set_xlabel('')
                    else:
                        ax.set_xlabel(nice_names[attribute])
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)

                if ax.is_first_col():
                    label = collection if i != 1 else 'MAP@10\n\n' + collection
                    ax.set_ylabel(label)
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)

        fig.tight_layout()
        fig.show()


def overall_parameters(df, baselines=False):
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
        current_axes.set_xlabel(nice_names[attribute])
    plt.show()


def impact_of_parameters(df):
    stds = pd.DataFrame(index=df.groupby('collection').groups, columns=attributes)
    for attribute in attributes:
        those = df.groupby(['collection', attribute]).mean()['MAP@10'].unstack()
        stds[attribute] = (those.max(axis=1) - those.min(axis=1)) / those.min(axis=1)
    return stds

