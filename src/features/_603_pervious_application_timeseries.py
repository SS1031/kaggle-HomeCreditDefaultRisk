import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import LatentDirichletAllocation

import const
from features import FeatherFeatureTrnTstDF
from features._999_utils import target_encode


class PreviousApplicationTimeseriesLDA(FeatherFeatureTrnTstDF):
    @staticmethod
    def input():
        return const.in_app_trn, const.in_app_tst

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(app_trn: pd.DataFrame, app_tst: pd.DataFrame,
                                       random_state: int):
        ncomp = 10

        prv = pd.read_feather('../data/input/previous_application.feather')

        ts_trn, ts_tst, ts_feat_cols = previous_application_timeseries(app_trn, app_tst, prv)

        lda = LatentDirichletAllocation(n_components=ncomp,
                                        random_state=random_state,
                                        n_jobs=-1)
        lda_trn = lda.fit_transform(ts_trn[ts_feat_cols])
        lda_tst = lda.transform(ts_tst[ts_feat_cols])

        lda_trn = pd.DataFrame(lda_trn, columns=["PRV_TS_LDA{}".format(i) for i in range(ncomp)])
        lda_tst = pd.DataFrame(lda_tst, columns=["PRV_TS_LDA{}".format(i) for i in range(ncomp)])

        lda_trn['SK_ID_CURR'] = app_trn.SK_ID_CURR
        lda_tst['SK_ID_CURR'] = app_tst.SK_ID_CURR

        return lda_trn, lda_tst


def previous_application_timeseries(app_train, app_test, prv):

    app_test.insert(1, 'TARGET', -1)

    trainsub = app_train[['SK_ID_CURR', 'TARGET']]
    testsub = app_test[['SK_ID_CURR', 'TARGET']]

    del app_train, app_test
    gc.collect()

    trainsub = trainsub.merge(prv, on='SK_ID_CURR', how='left')
    testsub = testsub.merge(prv, on='SK_ID_CURR', how='left')
    gc.collect()

    floattypes = []
    inttypes = []
    stringtypes = []
    for c in trainsub.columns[1:]:
        if (trainsub[c].dtype == 'object'):
            trainsub[c] = trainsub[c].astype('str')
            testsub[c] = testsub[c].astype('str')
            stringtypes.append(c)
        elif (trainsub[c].dtype == 'int64'):
            trainsub[c] = trainsub[c].astype('int32')
            testsub[c] = testsub[c].astype('int32')
            inttypes.append(c)
        else:
            trainsub[c] = trainsub[c].astype('float64')
            testsub[c] = testsub[c].astype('float64')
            floattypes.append(c)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for col in stringtypes:
        print(col)
        trainsub['te_' + col] = 0.
        testsub['te_' + col] = 0.
        SMOOTHING = testsub[~testsub[col].isin(trainsub[col])].shape[0] / testsub.shape[0]
        for f, (vis_index, blind_index) in enumerate(kf.split(trainsub, trainsub.TARGET)):

            _, trainsub.loc[blind_index, 'te_' + col] = target_encode(
                trainsub.loc[vis_index, col],
                trainsub.loc[blind_index, col],
                target=trainsub.loc[vis_index, 'TARGET'],
                min_samples_leaf=100,
                smoothing=SMOOTHING,
                noise_level=0.0
            )
            _, x = target_encode(
                trainsub.loc[vis_index, col],
                testsub[col],
                target=trainsub.loc[vis_index, 'TARGET'],
                min_samples_leaf=100,
                smoothing=SMOOTHING,
                noise_level=0.0)
            testsub['te_' + col] += (0.2 * x)

        trainsub.drop(col, inplace=True, axis=1)
        testsub.drop(col, inplace=True, axis=1)

    alldata = trainsub.append(testsub)
    del trainsub, testsub
    gc.collect()

    x = alldata['SK_ID_CURR'].value_counts().reset_index(drop=False)
    x.columns = ['SK_ID_CURR', 'cnt']

    alldata['SK_ID_PREV'] = alldata['SK_ID_PREV'].fillna(-1)

    alldata.sort_values(['SK_ID_CURR', 'SK_ID_PREV'], inplace=True, ascending=False)
    alldata['cc'] = alldata[['SK_ID_CURR']].groupby(['SK_ID_CURR']).cumcount()

    alldata = alldata.groupby(['SK_ID_CURR'])[alldata.columns].head(20)

    feats = [f for f in alldata.columns if f not in
             ['SK_ID_CURR', 'SK_ID_PREV', 'TARGET', 'cc']]

    # pivot_tableを一気にやるとメモリが爆発する
    # alldata = pd.pivot_table(alldata, index=['SK_ID_CURR', 'TARGET'],
    #                          values=feats, columns=['cc'])
    newdf = pd.DataFrame()
    for col in feats:
        print("Pivotting", col)
        partdf = pd.pivot_table(alldata, index=['SK_ID_CURR', 'TARGET'],
                                values=col, columns=['cc'])
        partdf.columns = [col + "_" + str(_c) for _c in partdf.columns]
        newdf = pd.concat([newdf, partdf], axis=1)

    alldata = newdf.copy()
    del newdf
    gc.collect()

    # alldata.columns = [
    #     x + "_" + str(y) for x, y in zip(alldata.columns.get_level_values(0),
    #                                      alldata.columns.get_level_values(1))
    # ]
    print("finish pivotting")

    alldata.reset_index(drop=False, inplace=True)
    alldata['nans'] = alldata.isnull().sum(axis=1)
    alldata = alldata.merge(x, on='SK_ID_CURR').fillna(0)

    feats = [f for f in alldata.columns if f not in [
        'SK_ID_CURR', 'SK_ID_PREV', 'TARGET', 'cc']]

    print("Scaling...")
    alldata[feats] -= alldata[feats].min()
    alldata[feats] /= alldata[feats].max()

    trainsub = alldata[alldata.TARGET != -1].copy().reset_index(drop=True)
    testsub = alldata[alldata.TARGET == -1].copy().reset_index(drop=True)
    testsub.drop(columns=['TARGET'], inplace=True)

    del alldata
    gc.collect()

    return trainsub, testsub, feats
