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


def map_files_to_df_components(file_string):
    mapping = {}
    for line in file_string.split('\n'):
        if not line:
            continue
        file = line.split()[-1]
        parts = file.split('/')
        mapping[parts[-2]] = parts[-4]

    df = pd.DataFrame(index=mapping.keys(), columns=['sub', 'win', 'epoch', 'min'])
    for k, v in mapping.items():
        split = v.split('-')
        d = {}
        for i, name in enumerate(split):
            if name in df.columns:
                d[name] = split[i + 1]
        df.loc[k] = d

    return df
