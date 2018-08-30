import feather
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


app_trn = feather.read_dataframe('../input/application_train.feather')


def target_bar(col, title):

    df0 = app_trn[app_trn["TARGET"] == 0]
    df1 = app_trn[app_trn["TARGET"] == 1]

    t0 = df0[col].value_counts().rename(col + '0')
    t1 = df1[col].value_counts().rename(col + '1')
    t = pd.concat([t0, t1], axis=1).fillna(0).astype(int)
    t['total'] = t.sum(axis=1)
    t.sort_values('total', inplace=True, ascending=False)
    t.drop(columns=['total'], inplace=True)

    idx = np.arange(len(t))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(idx, t[col + '0'], width)
    ax.bar(idx+width, t[col + '1'], width)

    ax.set_title('Scores by group and gender')
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels(t.index.values, rotation=45)

    plt.show()


target_bar('NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL')


def target_categorical_scatter(col, title):

    df0 = app_trn[app_trn["TARGET"] == 0]
    df1 = app_trn[app_trn["TARGET"] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.distplot(df0[col], ax=axes[0], color='b')
    axes[0].set_title('Target=0')
    sns.distplot(df1[col], ax=axes[1], color='orange')
    axes[1].set_title('Target=1')
    fig.suptitle(title)
    plt.show()


target_categorical_scatter('DAYS_BIRTH', 'DAY_BIRTH distribution by TARGET')
