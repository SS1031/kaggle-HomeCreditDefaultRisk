import gc
import pandas as pd
import numpy as np

import multiprocessing as mp

from scipy.stats import kurtosis, iqr, skew
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm


class GroupbyAggregateDiffs:

    def __init__(self, groupby_aggregations, use_diffs_only=False):

        self.groupby_aggregations = groupby_aggregations
        self.use_diffs_only = use_diffs_only

    def aggregate(self, main_table):

        features = []
        groupby_feature_names = []
        diff_feature_names = []

        print("Aggregation")
        for groupby_cols, specs in self.groupby_aggregations:
            group_object = main_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                group_features = group_object[select].agg(agg).reset_index().rename(
                    index=str,
                    columns={select: groupby_aggregate_name}
                )[groupby_cols + [groupby_aggregate_name]]
                features.append((groupby_cols, group_features))
                groupby_feature_names.append(groupby_aggregate_name)

        # Merge
        print("Merge")
        for groupby_cols, groupby_features in tqdm(features):
            main_table = main_table.merge(groupby_features, on=groupby_cols, how='left')

        # diffの計算
        print("Difference")
        for groupby_cols, specs in self.groupby_aggregations:
            for select, agg in specs:
                if agg in ['mean', 'median', 'max', 'min']:
                    groupby_aggregate_name = self._create_colname_from_specs(
                        groupby_cols, select, agg)
                    diff_feature_name = '{}_DIFF'.format(groupby_aggregate_name)
                    abs_diff_feature_name = '{}_ABS_DIFF'.format(groupby_aggregate_name)

                    main_table[diff_feature_name] = main_table[select] - \
                        main_table[groupby_aggregate_name]
                    main_table[abs_diff_feature_name] = np.abs(
                        main_table[select] - main_table[groupby_aggregate_name])

                    diff_feature_names.append(diff_feature_name)
                    diff_feature_names.append(abs_diff_feature_name)

        if self.use_diffs_only:
            return main_table[diff_feature_names]
        else:
            return main_table[groupby_feature_names + diff_feature_names]

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}-{}-{}'.format('.'.join(groupby_cols), select, agg.upper())


class GroupbyAggregate:

    def __init__(self, prefix, id_columns, groupby_aggregations):

        self.prefix = prefix
        self.id_columns = id_columns
        self.groupby_aggregations = groupby_aggregations

    def aggregate(self, table):
        features = pd.DataFrame({self.id_columns[0]: table[self.id_columns[0]].unique()})

        for groupby_cols, specs in self.groupby_aggregations:
            group_object = table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                features = features.merge(group_object[select]
                                          .agg(agg)
                                          .reset_index()
                                          .rename(index=str,
                                                  columns={select: groupby_aggregate_name})
                                          [groupby_cols + [groupby_aggregate_name]],
                                          on=groupby_cols,
                                          how='left')

        return features

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}-{}-{}-{}'.format(self.prefix, '_'.join(groupby_cols), select, agg.upper())


def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()

    return features


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def add_last_k_features_fractions(features, id, period_fractions):
    fraction_features = features[[id]].copy()

    for short_period, long_period in period_fractions:
        short_feature_names = get_feature_names_by_period(features, short_period)
        long_feature_names = get_feature_names_by_period(features, long_period)

        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk = '_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            fraction_features[fraction_feature_name] = features[short_feature] / \
                features[long_feature]
    return fraction_features


def get_feature_names_by_period(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def monthly_balance_timeseries_table(balance, scaling=True, period=10e10):
    floattypes = []
    inttypes = []
    for c in balance.columns:
        if c not in ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'TARGET']:
            if (balance[c].dtype == 'int64'):
                balance[c] = balance[c].astype('int32')
                inttypes.append(c)
            elif (balance[c].dtype == 'float64'):
                balance[c] = balance[c].astype('float64')
                floattypes.append(c)

    balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
    balance['cumcount'] = balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby(
        ['SK_ID_CURR', 'SK_ID_PREV']
    ).cumcount()

    feats = floattypes + inttypes
    pivotdf = pd.DataFrame()

    if period < 10e9:
        balance = balance[balance.cumcount < period].copy()

    for col in feats:
        print("Pivotting", col)
        partdf = pd.pivot_table(balance, index=['SK_ID_CURR', 'SK_ID_PREV'],
                                values=col, columns=['cumcount'])
        partdf.columns = [col + "_" + str(_c) for _c in partdf.columns]
        pivotdf = pd.concat([pivotdf, partdf], axis=1)
        del partdf
        gc.collect()

    timeseries_colnum_num = pivotdf.shape[1]
    pivotdf.fillna(-1, inplace=True)

    pivotdf.reset_index(inplace=True)
    pivotdf.drop(columns='SK_ID_PREV', inplace=True)
    pivotdf = pivotdf.groupby('SK_ID_CURR').mean()

    if scaling:
        print("Scaling...")
        pivotdf -= pivotdf.min()
        pivotdf /= pivotdf.max()

    print("Asserting...")
    assert pivotdf.shape[0] == balance.SK_ID_CURR.nunique()
    assert pivotdf.shape[1] == timeseries_colnum_num
    assert pivotdf.notnull().all().all()

    return pivotdf.reset_index()


def extract_knn_feature(X_trn, y_trn, X_tst, K, CLASS_NUM):
    for class_index in range(CLASS_NUM):

        clf = KNeighborsClassifier(metric='euclidean',
                                   leaf_size=200, n_jobs=-1)
        clf.fit(X_trn[y_trn == class_index], y_trn[y_trn == class_index])

        distances, index = clf.kneighbors(X_tst, n_neighbors=K, return_distance=True)

        if class_index == 0:
            knn_features = distances.cumsum(axis=1)
        else:
            knn_features = np.concatenate((knn_features, distances.cumsum(axis=1)),
                                          axis=1)
    return knn_features
