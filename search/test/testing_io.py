import datetime
import os

import pandas as pd


def print_table(data, args):
    data = combine_multirun(data)
    data = reorder_columns(data, args)
    pd.set_option('precision', args.precision)
    if args.latex:
        print(data.to_latex())
    else:
        print(data)


def display_chart(data, args):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    data = combine_multirun(data)
    data = reorder_columns(data, args)
    for row in data.index:
        plt.plot(data.loc[row], label=row)
    plt.legend()
    plt.xlabel(args.x_axis)
    plt.ylabel(args.y_axis or args.column)
    plt.title(args.title)
    plt.show()


def save_to_file(data, args):
    data = combine_multirun(data, grouping=False)
    save_file = args.save_file
    if not save_file:
        save_dir = os.path.join(os.getcwd(), 'output')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, datetime.datetime.now().isoformat() + ".pkl")
    print('Saving to file', save_file)
    data.to_pickle(save_file)


def reorder_columns(df, parsed_args):
    if parsed_args.column_order:
        try:
            return df[df.columns[list(map(int, list(parsed_args.column_order)))]]
        except:
            print('Unable to rearrange columns!')  # don't want to fail just cuz we can't rearrange columns
            return df
    else:
        cols = df.columns.tolist()
        try:
            cols = list(map(str, sorted(map(int, cols))))
        except ValueError:
            cols = sorted(cols)
        except TypeError:
            pass
        return df[cols]


def combine_multirun(results, grouping=True):
    if type(results) is not list:
        return results
    concat_results = pd.concat(results)
    return concat_results.astype(float).groupby(concat_results.index).mean() if grouping else concat_results