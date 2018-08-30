import gc
import numpy as np
import pandas as pd

import const

from features import FeatherFeatureIDFromDF
from features._000_cleaning import bureau_cleaning
from features._999_utils import GroupbyAggregate


class BureauAggregation(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_bur

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(bur: pd.DataFrame, random_state=None):
        # クリーニング
        bur = bureau_cleaning(bur)

        # 重要な特徴量をアクティブ or 非アクティブで集計したい
        bur['BUR_CREDIT_ACTIVE_BINARY'] = (bur.CREDIT_ACTIVE == 'Active').astype(int)
        bur['BUR_CREDIT_ACTIVE*DAYS_CREDIT'] = (bur.BUR_CREDIT_ACTIVE_BINARY * bur.DAYS_CREDIT)
        bur['BUR_CREDIT_ACTIVE*DAYS_CREDIT_ENDDATE'] = (bur.BUR_CREDIT_ACTIVE_BINARY * bur.DAYS_CREDIT_ENDDATE)

        # 集計前の変換
        bur['log(AMT_ANNUITY)'] = np.log(bur.AMT_ANNUITY + 0.001)
        bur['log(AMT_CREDIT_SUM)'] = np.log(bur.AMT_CREDIT_SUM + 0.001)
        bur['log(AMT_CREDIT_SUM)/log(AMT_ANNUITY)'] = \
            np.log(bur.AMT_CREDIT_SUM + 0.001) / np.log(bur.AMT_ANNUITY + 0.001)
        bur['log(AMT_CREDIT_SUM_DEBT)'] = np.log(bur.AMT_CREDIT_SUM_DEBT + 0.001)
        bur['log(AMT_CREDIT_SUM_LIMIT)'] = np.log(bur.AMT_CREDIT_SUM_LIMIT + 0.001)
        bur['log(AMT_CREDIT_SUM_DEBT)/log(AMT_CREDIT_SUM_LIMIT)'] = \
            np.log(bur.AMT_CREDIT_SUM_DEBT + 0.001) / np.log(bur.AMT_CREDIT_SUM_LIMIT + 0.001)
        bur['log(AMT_CREDIT_SUM_OVERDUE)'] = np.log(bur.AMT_CREDIT_SUM_OVERDUE + 0.001)
        bur['log(AMT_CREDIT_SUM_OVERDUE)/log(AMT_CREDIT_SUM_DEBT)'] = \
            np.log(bur.AMT_CREDIT_SUM_OVERDUE + 0.001) / np.log(bur.AMT_CREDIT_SUM_DEBT + 0.001)
        bur['log(AMT_CREDIT_MAX_OVERDUE)'] = np.log(bur.AMT_CREDIT_MAX_OVERDUE + 0.001)
        bur['log(CREDIT_DAY_OVERDUE)'] = np.log(bur.CREDIT_DAY_OVERDUE + 0.001)

        BUREAU_AGGREGATION_RECIPIES = [
            ('CREDIT_TYPE', 'count'),
            ('CREDIT_ACTIVE', 'size'),
            ('CNT_CREDIT_PROLONG', 'sum')
        ]

        for agg in ['mean', 'min', 'max', 'sum', 'var', 'skew']:
            for select in ['log(AMT_ANNUITY)',
                           'log(AMT_CREDIT_SUM)',
                           'log(AMT_CREDIT_SUM)/log(AMT_ANNUITY)',
                           'log(AMT_CREDIT_SUM_DEBT)',
                           'log(AMT_CREDIT_SUM_LIMIT)',
                           'log(AMT_CREDIT_SUM_DEBT)/log(AMT_CREDIT_SUM_LIMIT)',
                           'log(AMT_CREDIT_SUM_OVERDUE)',
                           'log(AMT_CREDIT_SUM_OVERDUE)/log(AMT_CREDIT_SUM_DEBT)',
                           'log(AMT_CREDIT_MAX_OVERDUE)',
                           'log(CREDIT_DAY_OVERDUE)',
                           'DAYS_CREDIT',
                           'DAYS_CREDIT_ENDDATE',
                           'DAYS_ENDDATE_FACT',
                           'DAYS_CREDIT_UPDATE']:
                BUREAU_AGGREGATION_RECIPIES.append((select, agg))
        BUREAU_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUREAU_AGGREGATION_RECIPIES)]
        feats = GroupbyAggregate('BUR', ('SK_ID_CURR', 'SK_ID_CURR'),
                                 BUREAU_AGGREGATION_RECIPIES).aggregate(bur)

        return feats


if __name__ == "__main__":
    feats = BureauAggregation(const.FEATDIR).create_feature(0)
