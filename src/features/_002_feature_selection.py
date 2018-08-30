import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import time
from lightgbm import LGBMClassifier
import lightgbm as lgb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings
import gc

warnings.simplefilter('ignore', UserWarning)
gc.enable()


def remove_high_correlation_column(train, test, threshold=0.9):
    print("Remove correlation rate over {}% columns".format(threshold * 100))

    # Absolute value correlation matrix
    corr_matrix = train.corr().abs()
    corr_matrix.head()

    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print('There are %d columns to remove.' % (len(to_drop)))

    train = train.drop(columns=to_drop)
    test = test.drop(columns=to_drop)

    print('Training set full shape: ', train.shape)
    print('Testing set full shape: ', test.shape)

    return train, test


def remove_high_missing_column(train, test, threshold=0.8):
    print("Remove missing rate over {}% columns".format(threshold * 100))

    train_missing = (train.isnull().sum() / len(train)).sort_values(ascending=False)
    print(train_missing.head())

    # test missing values (in percent)
    test_missing = (test.isnull().sum() / len(test)).sort_values(ascending=False)
    print(test_missing.head())

    # Identify missing values above threshold
    train_missing = train_missing.index[train_missing > threshold]
    test_missing = test_missing.index[test_missing > threshold]

    all_missing = list(set(set(train_missing) | set(test_missing)))

    print('There are {} columns with more than 80% missing values'.format(len(all_missing)))

    train = train.drop(columns=all_missing)
    test = test.drop(columns=all_missing)

    print('Training set full shape: ', train.shape)
    print('Testing set full shape: ', test.shape)

    return train, test


def plot_feature_importances(df, threshold=0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.

    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances

    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column

    """

    plt.rcParams['font.size'] = 18

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()

    # Cumulative importance plot
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.show()

    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' %
          (importance_index + 1, threshold))

    return df


def remove_low_importance_feature(trn, target, tst, iterations=2, threshold=5):
    """
    Identify zero importance features in a training dataset based on the
    feature importances from a gradient boosting model.

    Parameters
    --------
    train : dataframe
        Training features

    train_labels : np.array
        Labels for training data

    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """

    print("Remove 0 lgb importance columns")

    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(trn.shape[1])

    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):
        print("LGB importance iteration No.{}".format(i + 1))
        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(
            trn,
            target,
            test_size=0.25,
            random_state=i * 123,
        )

        d_train = lgb.Dataset(train_features, label=train_y)
        d_valid = lgb.Dataset(valid_features, label=valid_y)

        # Train using early stopping
        eval_results = {}
        model = lgb.train(params={"boosting_type": "gbdt",
                                  "objective": "binary",
                                  "metric": "auc",
                                  },
                          train_set=d_train,
                          valid_sets=[d_train, d_valid],
                          valid_names=['train', 'valid'],
                          verbose_eval=100,
                          evals_result=eval_results)

        # Record the feature importances
        feature_importances += model.feature_importance() / iterations

    feature_importances = pd.DataFrame(
        {'feature': list(trn.columns), 'importance': feature_importances}
    ).sort_values('importance', ascending=False)

    # Find the features with zero importance
    low_features = list(
        feature_importances[feature_importances['importance'] < threshold]['feature']
    )

    print('\nThere are {} features under {} importance'.format(len(low_features), threshold))

    trn = trn.drop(columns=low_features)
    tst = tst.drop(columns=low_features)

    print('Training set full shape: ', trn.shape)
    print('Testing set full shape: ', tst.shape)

    return trn, tst


def get_feature_importances(data, target, shuffle, seed=None):
    # Gather real features
    train_features = data.columns.tolist()

    # Go over fold and keep track of CV score (train and valid) and feature importances
    # Shuffle target if required
    # y = data['TARGET'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        target = target.copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data, target, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(target, clf.predict(data))

    return imp_df


def build_null_importances_distribution(data, target):
    null_imp_df = pd.DataFrame()
    nb_runs = 10
    import time
    start = time.time()
    dsp = ''
    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(data, target, shuffle=True)
        imp_df['run'] = i + 1
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=True)
        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)

    return null_imp_df


def feature_socring(actual_imp_df, null_imp_df):
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(
            1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(
            1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    return scores_df


def corr_scoring(actual_imp_df, null_imp_df):
    correlation_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores,
                                      columns=['feature', 'split_score', 'gain_score'])

    return corr_scores_df, correlation_scores


def plot_feature_score(scores_df):
    plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70],
                ax=ax)
    ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70],
                ax=ax)
    ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_corr_score(corr_scores_df):
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    sns.barplot(x='split_score', y='feature',
                data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
    ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:70],
                ax=ax)
    ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
    fig.subplots_adjust(top=0.93)
    plt.show()


def score_feature_selection(df=None, train_features=None, target=None):
    # Fit LightGBM
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': .1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 13,
        'min_split_gain': .00001,
        'reg_alpha': .00001,
        'reg_lambda': .00001,
        'metric': 'auc'
    }

    # Fit the model
    hist = lgb.cv(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=2000,
        nfold=5,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=0,
        seed=17
    )
    # Return the last mean / std values
    return hist['auc-mean'][-1], hist['auc-stdv'][-1]


def null_importances_feature_selection(trn, target, tst, do_plot=False):
    print("Start null importances feature selection...")
    # https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
    # 基本的なアイディア
    #   1. null importances distribution の作成
    #       ターゲットをランダムシャッフルしてlgbmを複数回訓練、特徴量重要度を算出
    #       この分布がその特徴量がtargetについてどれだけ関係しているか示す
    #   2. オリジナルのtargetを使ってfeature importanceを算出、null importances distributionと比較

    actual_imp_df = get_feature_importances(trn, target, shuffle=False)
    null_imp_df = build_null_importances_distribution(trn, target)
    scores_df = feature_socring(actual_imp_df, null_imp_df)
    corr_scores_df, correlation_scores = corr_scoring(actual_imp_df, null_imp_df)
    if do_plot:
        plot_feature_score(scores_df)
        plot_corr_score(corr_scores_df)

    results = []
    feats = []
    for threshold in [0, 10, 20, 30, 40, 50, 60, 70]:
        split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
        gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]

        print('Results for threshold %3d' % threshold)
        split_results = score_feature_selection(df=trn, train_features=split_feats,
                                                target=target)
        print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
        gain_results = score_feature_selection(df=trn, train_features=gain_feats,
                                               target=target)
        print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))

        results.append([split_results[0], gain_results[0]])
        feats.append([split_feats, gain_feats])

    bst_feats = [__f for _f in feats for __f in _f][np.argmax(results)]

    trn = trn[bst_feats].copy()
    tst = tst[bst_feats].copy()

    print('Training set shape after selection: ', trn.shape)
    print('Testing set shape after selection : ', tst.shape)

    return trn, tst


if __name__ == "__main__":
    import os
    import sys

    sys.path.append('./')

    import json
    from collections import OrderedDict
    from features._001_load import datasets

    configs = json.load(open('./configs/lightgbm43.json'), object_pairs_hook=OrderedDict)
    conf = configs['config_set'][0]
    trn, tst, categorical_features = datasets(conf['feature'],
                                              random_state=conf['random_seed'],
                                              debug=False)
    target = trn.TARGET.copy()

    trn_sk_id_curr = trn.SK_ID_CURR.copy()
    tst_sk_id_curr = tst.SK_ID_CURR.copy()

    trn.drop(columns=['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    tst.drop(columns=['SK_ID_CURR'], axis=1, inplace=True)

    trn, tst = null_importances_feature_selection(trn, target, tst, do_plot=True)
