import gc
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import const
from features import FeatherFeatureTrnTstDF
from features._000_cleaning import application_cleaning
from features._999_utils import extract_knn_feature


class ApplicationEXTSourceKNN(FeatherFeatureTrnTstDF):

    @staticmethod
    def input():
        return const.in_app_trn, const.in_app_tst

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(trn: pd.DataFrame, tst: pd.DataFrame, random_state: int):
        y = trn.TARGET

        trn.drop(columns='TARGET', inplace=True)

        trn['is_tst'] = False
        tst['is_tst'] = True

        trn_sk_id_curr = trn.SK_ID_CURR.copy()
        tst_sk_id_curr = tst.SK_ID_CURR.copy()
        trn_tst = pd.concat([trn, tst], axis=0).reset_index(drop=True)

        del trn, tst
        gc.collect()

        trn_tst = application_cleaning(trn_tst)
        trn_tst = trn_tst[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'is_tst']]

        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            trn_tst['EXT_SOURCES_{}'.format(function_name)] = eval(
                'np.{}'.format(function_name)
            )(trn_tst[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        # EXTERNAL_SOURCEを重み付けで加算
        trn_tst['EXT_SOURCES_WEIGHTED'] = (
                (trn_tst.EXT_SOURCE_1 * 2 + trn_tst.EXT_SOURCE_2 * 3 + trn_tst.EXT_SOURCE_3 * 4) / 9
        )

        feature_cols = [c for c in trn_tst.columns if c != 'is_tst']
        print("NUM OF FEATURE:", len(feature_cols))

        trn_tst[feature_cols] = trn_tst[feature_cols].fillna(-1)
        trn_tst[feature_cols] = preprocessing.MinMaxScaler().fit_transform(trn_tst[feature_cols])

        trn = trn_tst.query(
            'is_tst == False'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        tst = trn_tst.query(
            'is_tst == True'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        del trn_tst
        gc.collect()

        K = 2
        CLASS_NUM = len(set(y))
        trn_knn = np.zeros((len(trn), CLASS_NUM * K))
        tst_knn = np.zeros((len(tst), CLASS_NUM * K))

        num_fold = 5
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trn, y)):
            print("Fold", n_fold)
            X_trn, X_val, y_trn = (trn.iloc[trn_idx], trn.iloc[val_idx], y.iloc[trn_idx])
            trn_knn[val_idx] = extract_knn_feature(X_trn, y_trn, X_val, K, CLASS_NUM)
            tst_knn += extract_knn_feature(X_trn, y_trn, tst, K, CLASS_NUM) / num_fold

        trn_knn = pd.DataFrame(
            trn_knn, columns=['APP_EXT_KNN{}'.format(i + 1) for i in range(trn_knn.shape[1])]
        )
        tst_knn = pd.DataFrame(
            tst_knn, columns=['APP_EXT_KNN{}'.format(i + 1) for i in range(tst_knn.shape[1])]
        )

        trn_knn['SK_ID_CURR'] = trn_sk_id_curr
        tst_knn['SK_ID_CURR'] = tst_sk_id_curr

        return trn_knn, tst_knn


if __name__ == '__main__':
    app_ext_knn_trn, app_ext_knn_tst = ApplicationEXTSourceKNN(const.FEATDIR).create_feature(0)
