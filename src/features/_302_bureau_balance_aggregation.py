import os

import pandas as pd
import numpy as np
import gc

import const
from features import FeatherFeatureIDFromDF


class BureauBalanceAggregation(FeatherFeatureIDFromDF):

    """
    bureau_balanceの集計
    """
    @staticmethod
    def input():
        return const.in_bur

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(bur: pd.DataFrame, random_state=None):

        bbl = pd.read_feather(const.in_bbl)

        bbl_status_dum = pd.get_dummies(bbl.STATUS, dummy_na=True)
        bbl_status_dum.columns = ['STATUS_' +
                                  str(col) for col in bbl_status_dum.columns]

        bbl = pd.concat([bbl, bbl_status_dum], axis=1)

        bbl_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bbl_status_dum.columns:
            bbl_aggregations[col] = ['mean']

        bbl_agg = bbl.groupby('SK_ID_BUREAU').agg(bbl_aggregations)
        bbl_agg.columns = ["BBL_" + e[0] + "_" + e[1].upper() for e in bbl_agg.columns.tolist()]
        bbl_agg.reset_index(inplace=True)

        bbl = pd.merge(bur[['SK_ID_CURR', 'SK_ID_BUREAU']], bbl_agg, on='SK_ID_BUREAU', how='left')

        # burからのaggregation
        bur_aggregations = {'BBL_MONTHS_BALANCE_SIZE': ['mean', 'max', 'min'],
                            'BBL_MONTHS_BALANCE_MIN': ['mean', 'max', 'min'],
                            'BBL_MONTHS_BALANCE_MAX': ['mean', 'max', 'min']}
        for col in [col for col in bbl_agg.columns if 'STATUS' in col]:
            bur_aggregations[col] = ['mean']

        feats = bbl.groupby('SK_ID_CURR')[
            [col for col in bbl.columns if col not in ['SK_ID_BUREAU', 'SK_ID_CURR']]
        ].agg(bur_aggregations)
        feats.columns = [e[0] + "_" + e[1].upper() for e in feats.columns.tolist()]

        return feats.reset_index()


if __name__ == "__main__":

    feat_path = BureauBlanceAggregation(const.FEATDIR).create_feature(0)
