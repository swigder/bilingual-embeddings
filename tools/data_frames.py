import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import gridspec


SCORE = 'MAP@10'
COLLECTION = 'collection'
PRETRAINED = 'pretrained'
EPOCHS = 'epochs'
WINDOW_SIZE = 'win'
MIN_SUBWORD = 'sub'
NORMALIZE = 'norm'
USE_SUBWORD = 'subword'

attributes = [MIN_SUBWORD, WINDOW_SIZE, EPOCHS, NORMALIZE, USE_SUBWORD, PRETRAINED]
nice_names = {MIN_SUBWORD: 'Minimum subword length',
              WINDOW_SIZE: 'Window size',
              EPOCHS: 'Epochs',
              NORMALIZE: 'Normalize length',
              USE_SUBWORD: 'Use subword for OOV',
              PRETRAINED: 'Model type'}
baseline_columns = {MIN_SUBWORD: 'No',
                    WINDOW_SIZE: 5,
                    EPOCHS: 5,
                    NORMALIZE: False,
                    USE_SUBWORD: False,
                    PRETRAINED: 'Collection'}
replacements = {COLLECTION: {'ohsu-trec': 'ohsu'},
                MIN_SUBWORD: {'7': 'No'},
                PRETRAINED: {True: 'Hybrid', False: 'Collection'}}
collections = ['adi', 'time', 'ohsu']


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

    df = pd.DataFrame(index=mapping.keys(), columns=[COLLECTION, MIN_SUBWORD, WINDOW_SIZE, 'epoch', 'min'])
    for k, v in mapping.items():
        split = v.split('-')
        d = {COLLECTION: '-'.join(split[:3])}
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
        assert norm == 'nn' or norm == NORMALIZE
        assert subword == USE_SUBWORD or subword == 'zero'
        norm = norm == NORMALIZE
        subword = subword == USE_SUBWORD
    columns = [MIN_SUBWORD, WINDOW_SIZE, EPOCHS, PRETRAINED]
    if not contains_all:
        columns += [COLLECTION, NORMALIZE, USE_SUBWORD]
    hyperparams = pd.DataFrame(index=results.index, columns=columns)
    for result in hyperparams.index:
        file_name = result if not contains_all else results.at[result, 'embedding']
        parts = file_name_only(file_name).split('-')
        assert parts[0] == collection
        d = {PRETRAINED: pretrained}
        d.update({COLLECTION: collection, NORMALIZE: norm, USE_SUBWORD: subword} if not contains_all else {})
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
    return df.loc[df.groupby(COLLECTION)[SCORE].idxmax()]


def plot_per_collection_single(df, plot_fn, suptitle, combine_legend=True):
    sns.set()
    sns.set_palette(['#66c2a5', '#fc8d62', '#8da0cb'])

    fig = plt.figure(figsize=(9, 3))
    # fig.suptitle(suptitle, y=.99)
    axes = fig.subplots(1, 3)

    collection_groups = df.groupby(COLLECTION)
    for i, collection in enumerate(collections):
        collection_df = collection_groups.get_group(collection)
        ax = axes[i]

        plot_fn(collection_df, ax)
        ax.set_title(collection)
        if ax.get_xlabel() in nice_names:
            ax.set_xlabel(nice_names[ax.get_xlabel()])

        if not ax.is_first_col():
            ax.set_ylabel('')

    if combine_legend:
        for ax in axes[:-1]:
            if ax.legend_:
                ax.legend_.remove()
        if axes[-1].get_legend():
            plt.setp(axes[-1].get_legend().get_texts(), fontsize='x-small')
            plt.setp(axes[-1].get_legend().get_title(), fontsize='x-small')
            axes[-1].legend_.set_bbox_to_anchor((1, 1.05))

    fig.tight_layout()
    fig.show()


def plot_per_collection(df, columns, plot_fn, suptitle, share_x=False, column_labels=None, x_label=True):
    if not column_labels:
        column_labels = {column: column for column in columns}

    sns.set()
    sns.set_palette(['#66c2a5', '#fc8d62', '#8da0cb'])

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(suptitle)
    outer = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.1)

    collection_groups = df.groupby(COLLECTION)
    for i, collection in enumerate(collections):
        collection_df = collection_groups.get_group(collection)

        inner = gridspec.GridSpecFromSubplotSpec(1, len(columns), subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        axes = np.empty(shape=(1, len(columns)), dtype=object)
        for j in range(len(columns)):
            sharex_param = {'sharex': axes[0, 0]} if share_x else {}
            ax = plt.Subplot(fig, inner[j], **sharex_param, sharey=axes[0, 0])
            fig.add_subplot(ax)
            axes[0, j] = ax

        for j, column in enumerate(columns):
            ax = axes[0, j]
            grid = plot_fn(collection_df, column, ax)
            if x_label:
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


def plot_all_parameter_interaction(df, parameters=attributes, reverse=False):
    if type(parameters) is str:
        parameters = [parameters]
    for attribute in parameters:
        others = [a for a in attributes if a != attribute]

        def df_function(collection_df, other, ax):
            x_attr, hue_attr = (other, attribute) if not reverse else (attribute, other)
            sns.stripplot(x=x_attr, y=SCORE, hue=hue_attr, data=collection_df,
                          order=sorted(collection_df[x_attr].unique()),
                          jitter=0.1, dodge=True, alpha=0.5, ax=ax)

        plot_per_collection(df, others, df_function, nice_names[attribute],
                            share_x=False, column_labels=defaultdict(lambda: nice_names[attribute]), x_label=False)


def plot_single_parameter(df, attribute, split=PRETRAINED):
    def df_function(collection_df, ax):
        hue_order_option = {'hue_order': sorted(collection_df[split].unique())} if split else {}
        sns.stripplot(x=attribute, y=SCORE, hue=split, data=collection_df,
                      order=sorted(collection_df[attribute].unique()),
                      **hue_order_option,
                      jitter=1, dodge=True, alpha=0.5, ax=ax)
    plot_per_collection_single(df, df_function, nice_names[attribute])


def plot_overall_parameters(df, split=PRETRAINED):
    others = [a for a in attributes if a != split]

    def df_function(collection_df, attribute, ax):
        sns.stripplot(x=attribute, y=SCORE, hue=split, data=collection_df,
                      order=sorted(collection_df[attribute].unique()),
                      jitter=1, dodge=True, alpha=0.5, ax=ax)
        # sns.pointplot(x=attribute, y=SCORE, hue=split, data=collection_df,
        #               order=sorted(collection_df[attribute].unique()),
        #               dodge=0.01, join=False, markers="d", scale=1, ax=ax)

    plot_per_collection(df, others, df_function, 'Overall paramater impact',
                        share_x=False, column_labels=nice_names, x_label=False)


DIFF_ABS = 'Change in MAP@10'
DIFF_REL = 'Change in MAP@10 (Rel)'
TYPE = 'type'


def _parameter_change_df(df, attribute, single_jump=False):
    new_index = list(set(df.columns).difference([attribute, SCORE]))
    values = sorted(df[attribute].unique())
    if single_jump:
        values = [values[0], values[-1]]
    columns = [(values[i-1], values[i]) for i in range(1, len(values))]
    column_names = ['{} to {}'.format(i, j) for i, j in columns]

    indexed = df.set_index(new_index)
    difference_dfs = []
    for (before, after), column_name in zip(columns, column_names):
        difference_df = pd.DataFrame(index=indexed.index, columns=[DIFF_ABS, DIFF_REL, TYPE])
        before_df = indexed[indexed[attribute] == before][SCORE]
        after_df = indexed[indexed[attribute] == after][SCORE]
        difference_df[DIFF_ABS] = after_df - before_df
        difference_df[DIFF_REL] = (after_df - before_df) / before_df
        difference_df[TYPE] = column_name
        difference_df.reset_index(inplace=True)
        difference_dfs.append(difference_df)

    return pd.concat(difference_dfs)


def plot_single_parameter_change(df, attribute, split=PRETRAINED, relative=False):
    diff_col = DIFF_REL if relative else DIFF_ABS

    def df_function(collection_df, ax):
        order_option = {'order': sorted(collection_df[split].unique())} if split is not None else {}
        sns.stripplot(x=split or TYPE, y=diff_col, hue=TYPE, data=collection_df,
                      **order_option,
                      jitter=1, dodge=True, alpha=0.2, ax=ax)
        sns.pointplot(x=split or TYPE, y=diff_col, hue=TYPE, data=collection_df,
                      **order_option,
                      dodge=False, join=False, markers='d', scale=1, ax=ax, legend=True)
        ymin, ymax = collection_df[diff_col].min(), collection_df[diff_col].max()
        ratio = 2  # of positive to negative
        ymin, ymax = (ymax / -ratio, ymax) if abs(ymax) > abs(ymin * ratio) else (ymin, ymin * -ratio)
        ax.set_ylim(bottom=ymin * 1.2, top=ymax * 1.2)
        handles, labels = ax.get_legend_handles_labels()
        legend_start = len(handles) // 2
        ax.legend(handles[legend_start:], labels[legend_start:], title=nice_names[attribute])

    plot_per_collection_single(df, df_function, nice_names[attribute], combine_legend=True)


def single_parameter_change(df, attribute, split=PRETRAINED, single_jump=False):
    change_df = _parameter_change_df(df, attribute, single_jump=single_jump)

    plot_single_parameter_change(change_df, attribute, split, relative=False)

    groupby = [COLLECTION, TYPE] if not split else [COLLECTION, split, TYPE]
    grouped = change_df.reset_index().groupby(groupby)
    print(grouped.agg({DIFF_ABS: ['mean', 'min', 'max'], DIFF_REL: ['mean']})[[DIFF_ABS, DIFF_REL]])
    print('max\n', grouped.max())
    print('min\n', grouped.min())
