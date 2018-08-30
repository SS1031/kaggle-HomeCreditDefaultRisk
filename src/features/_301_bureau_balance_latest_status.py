import os

import pandas as pd
import numpy as np
import gc

import const
from features import FeatherFeatureIDFromDF


class BureauBlanceLatestStatusNorm(FeatherFeatureIDFromDF):
    """
    bureau_balanceのSK_ID_BUREAUにおける最新STATUSのNorm
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
        piv_bbl = bbl.pivot(index='SK_ID_BUREAU',
                            columns='MONTHS_BALANCE',
                            values='STATUS')

        bbl_latest_status = piv_bbl[[0]].reset_index()
        bbl_latest_status.columns = ['SK_ID_BUREAU', 'BBL_LATEST_STATUS']

        bur = bur.merge(bbl_latest_status, on='SK_ID_BUREAU', how='left')
        del bbl_latest_status
        gc.collect()

        bbl_latest_status_dum = pd.get_dummies(bur.BBL_LATEST_STATUS, dummy_na=True)
        bbl_latest_status_dum.columns = ['BBL_LATEST_STATUS_' +
                                         str(col) for col in bbl_latest_status_dum.columns]

        bur = pd.concat([bur[['SK_ID_CURR']], bbl_latest_status_dum], axis=1)

        return bur.groupby('SK_ID_CURR').mean().reset_index()


if __name__ == "__main__":
    bbl_latest_status_norm = BureauBlanceLatestStatusNorm(const.FEATDIR).create_feature(0)

