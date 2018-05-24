import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import gridspec


SCORE = 'MAP@10'

attributes = ['sub', 'win', 'epochs', 'norm', 'subword', 'pretrained']
nice_names = {'sub': 'Minimum subword length',
              'win': 'Window size',
              'epochs': 'Epochs',
              'norm': 'Normalize length',
              'subword': 'Use subword for OOV',
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


def map_files_to_df_components(file_path):
    mapping = {}
    with open(file_path) as f:
        for line in f:
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
    return df.drop_duplicates(df.columns.difference([SCORE]))


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


def get_results(file_path, prefix, files):
    r = pd.concat([map_files_to_df_components(file_path), unpickle_multiple(prefix, files)], axis=1)
    return r.apply(pd.to_numeric, errors='ignore')


def get_max_map(df):
    return df.loc[df.groupby('collection')[SCORE].idxmax()]


def group_scores(df, groupby, average, n=2):
    if average:
        return df.groupby(groupby)[SCORE].mean().unstack()
    else:
        return df.sort_values([SCORE], ascending=False).groupby(groupby).head(n).groupby(
            groupby).min()[SCORE].unstack()


def plot_per_collection(df, columns, df_function, suptitle, share_x=False, column_labels=None):
    if not column_labels:
        column_labels = {column: column for column in columns}

    sns.set()

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(suptitle)
    outer = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.1)

    collection_groups = df.groupby('collection')
    for i, collection in enumerate(collection_groups.groups):
        collection_df = collection_groups.get_group(collection)

        inner = gridspec.GridSpecFromSubplotSpec(1, len(columns), subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        axes = np.empty(shape=(1, len(columns)), dtype=object)
        for j in range(len(columns)):
            sharex_param = {'sharex': axes[0, 0]} if share_x else {}
            ax = plt.Subplot(fig, inner[j], **sharex_param, sharey=axes[0, 0])
            fig.add_subplot(ax)
            axes[0, j] = ax

        for j, column in enumerate(columns):
            grid = df_function(collection_df, column)
            ax = axes[0, j]
            grid.plot(ax=ax)
            values = grid.index.tolist()
            if ax.is_last_row():
                ax.set_xticks(values if all(type(x) is int for x in values) else range(len(values)))
                ax.set_xticklabels(values)
                if share_x and j != len(columns) // 2:
                    ax.set_xlabel('')
                else:
                    ax.set_xlabel(column_labels[column])
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

            if ax.is_first_col():
                label = collection if i != 1 else 'MAP@10\n\n' + collection
                ax.set_ylabel(label)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

    fig.tight_layout()
    fig.show()


def parameter_interaction(df, options, parameters=attributes):
    for attribute in parameters:
        others = [a for a in attributes if a != attribute]

        def df_function(collection_df, other):
            return group_scores(collection_df, [attribute, other], **options)

        plot_per_collection(df, others, df_function, nice_names[attribute],
                            share_x=True, column_labels=defaultdict(lambda: nice_names[attribute]))


def overall_parameters(df, options, baselines=False, split='pretrained'):
    others = [a for a in attributes if a != split]

    def df_function(collection_df, other):
        a = group_scores(collection_df, [other, split], **options)
        if baselines:
            baseline_values = df.where(df[other] == baseline_columns[other]).groupby('collection').mean()['MAP@10']
            a = (a - baseline_values) / baseline_values
        return a

    plot_per_collection(df, others, df_function, 'Overall paramater impact',
                        share_x=False, column_labels=nice_names)


def impact_of_parameters(df, options, split='collection'):
    stds = pd.DataFrame(index=df.groupby(split).groups, columns=attributes)
    for attribute in attributes:
        those = group_scores(df, [attribute, split], **options).transpose()
        stds[attribute] = (those.max(axis=1) - those.min(axis=1)) / those.min(axis=1)
    return stds

