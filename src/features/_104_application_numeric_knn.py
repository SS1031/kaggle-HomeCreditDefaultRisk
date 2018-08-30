import gc
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import const
from features import FeatherFeatureTrnTstDF
from features._000_cleaning import application_cleaning
from features._999_utils import extract_knn_feature


class ApplicationNumericKNN(FeatherFeatureTrnTstDF):
    """
    ↓のkernelで紹介されている"good tranfomations"を活用してい
    https://www.kaggle.com/returnofsputnik/good-transformations-to-continuous-variables
    """

    @staticmethod
    def input():
        return const.in_app_trn, const.in_app_tst

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(trn: pd.DataFrame, tst: pd.DataFrame, random_state: int):

        y = trn.TARGET.copy()
        trn.drop(columns='TARGET', inplace=True)

        trn['is_tst'] = False
        tst['is_tst'] = True

        trn_sk_id_curr = trn.SK_ID_CURR.copy()
        tst_sk_id_curr = tst.SK_ID_CURR.copy()
        trn_tst = pd.concat([trn, tst], axis=0).reset_index(drop=True)

        del trn, tst
        gc.collect()

        trn_tst = application_cleaning(trn_tst)

        trans = pd.DataFrame()
        trans['log1p(AMT_INCOME_TOTAL)'] = np.log1p(trn_tst.AMT_INCOME_TOTAL)
        trans['log1p(AMT_CREDIT)'] = np.log1p(trn_tst.AMT_CREDIT)
        trans['log1p(AMT_ANNUITY)'] = np.log1p(trn_tst.AMT_ANNUITY)
        trans['AMT_CREDIT/AMT_ANNUITY'] = trn_tst.AMT_CREDIT / trn_tst.AMT_ANNUITY
        trans['log1p(AMT_GOODS_PRICE)'] = np.log1p(trn_tst.AMT_GOODS_PRICE)
        trans['sqrt(REGION_POPULATION_RELATIVE)'] = np.sqrt(trn_tst.REGION_POPULATION_RELATIVE)
        trans['sqrt(abs(DAYS_BIRTH))'] = np.sqrt(np.abs(trn_tst.DAYS_BIRTH))
        trans['sqrt(abs(DAYS_EMPLOYED))'] = np.sqrt(np.abs(trn_tst.DAYS_EMPLOYED))
        trans['sqrt(abs(DAYS_REGISTRATION))'] = np.sqrt(np.abs(trn_tst.DAYS_REGISTRATION))
        trans['sqrt(abs(OWN_CAR_AGE))'] = np.sqrt(np.abs(trn_tst.OWN_CAR_AGE))
        trans['log1p(APARTMENTS_AVG*50)'] = np.log1p(trn_tst.APARTMENTS_AVG * 50)
        trans['YEARS_BEGINEXPLUATATION_AVG^30'] = trn_tst.YEARS_BEGINEXPLUATATION_AVG ** 30
        trans['YEARS_BUILD_AVG^3'] = trn_tst.YEARS_BUILD_AVG ** 3
        trans['(ELEVATORS_AVG)^(1/40)'] = trn_tst.ELEVATORS_AVG ** (1/40)
        trans['(ENTRANCES_AVG)^(1/3)'] = trn_tst.ENTRANCES_AVG ** (1/40)
        trans['(FLOORSMAX_AVG)^(1/2.5)'] = trn_tst.FLOORSMAX_AVG ** (1/2.5)
        trans['(LIVINGAPARTMENTS_AVG)^(1/3)'] = trn_tst.LIVINGAPARTMENTS_AVG ** (1/3)
        trans['(LIVINGAREA_AVG)^(1/3)'] = trn_tst.LIVINGAREA_AVG ** (1/3)
        trans['(NONLIVINGAREA_AVG)^(1/5)'] = trn_tst.NONLIVINGAREA_AVG ** (1/5)
        trans['(TOTALAREA_MODE)^(1/3)'] = trn_tst.NONLIVINGAREA_AVG ** (1/3)
        trans['(OBS_30_CNT_SOCIAL_CIRCLE)^(1/7)'] = trn_tst.OBS_30_CNT_SOCIAL_CIRCLE ** (1/7)
        trans['(DEF_30_CNT_SOCIAL_CIRCLE)^(1/7)'] = trn_tst.DEF_30_CNT_SOCIAL_CIRCLE ** (1/7)

        trans['is_tst'] = trn_tst.is_tst

        feature_cols = [c for c in trans.columns if c != 'is_tst']
        trans[feature_cols] = trans[feature_cols].fillna(-1)
        trans[feature_cols] = preprocessing.MinMaxScaler().fit_transform(trans[feature_cols])

        trn = trans.query(
            'is_tst == False'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        tst = trans.query(
            'is_tst == True'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        del trans
        gc.collect()

        K = 2
        CLASS_NUM = len(set(y))
        trn_knn = np.zeros((len(trn), CLASS_NUM * K))
        tst_knn = np.zeros((len(tst), CLASS_NUM * K))

        print("NUM OF FEATURE:", len(feature_cols))

        num_fold = 5
        folds = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=random_state)
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trn, y)):
            print("Fold", n_fold)
            X_trn, X_val, y_trn = (trn.iloc[trn_idx], trn.iloc[val_idx], y.iloc[trn_idx])
            trn_knn[val_idx] = extract_knn_feature(X_trn, y_trn, X_val, K, CLASS_NUM)
            tst_knn += extract_knn_feature(X_trn, y_trn, tst, K, CLASS_NUM) / num_fold

        trn_knn = pd.DataFrame(
            trn_knn, columns=['APP_NUM_KNN{}'.format(i + 1) for i in range(trn_knn.shape[1])]
        )
        tst_knn = pd.DataFrame(
            tst_knn, columns=['APP_NUM_KNN{}'.format(i + 1) for i in range(tst_knn.shape[1])]
        )

        trn_knn['SK_ID_CURR'] = trn_sk_id_curr
        tst_knn['SK_ID_CURR'] = tst_sk_id_curr

        return trn_knn, tst_knn


def target_hist(df, col):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df0 = df[df["TARGET"] == 0]
    df1 = df[df["TARGET"] == 1]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.distplot(df0[col].dropna(), ax=ax, color='b')
    ax.set_title(col)
    sns.distplot(df1[col].dropna(), ax=ax, color='orange')
    plt.show()


if __name__ == '__main__':
    trn_knn_path, tst_knn_path = ApplicationNumericFeatsKNN(const.FEATDIR).create_feature(0)
