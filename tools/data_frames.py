import pandas as pd


def load_and_combine(path):
    df = pd.read_pickle(path)
    df = df.astype(float).groupby(df.index).mean()
    cols = list(map(str, sorted(map(int, df.columns.tolist()))))
    df = df[cols]
    return df


def add_percentage_change(df, from_col, to_col):
    df['Change'] = (df[to_col] - df[from_col]) / df[from_col] * 100


def load_change_show(path, from_col, to_col):
    df = load_and_combine(path)
    add_percentage_change(df, from_col, to_col)
    pd.set_option('precision', 4)
    print(df.to_latex())
    return df
