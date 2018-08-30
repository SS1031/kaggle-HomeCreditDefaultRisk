import pandas as pd

import const

from features import FeatherFeatureIDFromDF
from features._000_cleaning import previous_application_cleaning
from features._999_utils import GroupbyAggregate


class PreviousApplicationAggregation(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_prv

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(prv: pd.DataFrame, random_state: int):

        prv = previous_application_cleaning(prv)
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['AMT_ANNUITY',
                           'AMT_APPLICATION',
                           'AMT_CREDIT',
                           'AMT_DOWN_PAYMENT',
                           'AMT_GOODS_PRICE',
                           'CNT_PAYMENT',
                           'DAYS_DECISION',
                           'HOUR_APPR_PROCESS_START',
                           'RATE_DOWN_PAYMENT']:
                PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))

        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [
            (['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)
        ]

        prv_approved = prv[prv.NAME_CONTRACT_STATUS == 'Approved']
        prv_refused = prv[prv.NAME_CONTRACT_STATUS == 'Refused']

        feats_all = GroupbyAggregate(
            'PRV', ('SK_ID_CURR', 'SK_ID_CURR'),
            PREVIOUS_APPLICATION_AGGREGATION_RECIPIES
        ).aggregate(prv).set_index('SK_ID_CURR')

        feats_approved = GroupbyAggregate(
            'PRV_APPROVED', ('SK_ID_CURR', 'SK_ID_CURR'),
            PREVIOUS_APPLICATION_AGGREGATION_RECIPIES
        ).aggregate(prv_approved).set_index('SK_ID_CURR')

        feats_refused = GroupbyAggregate(
            'PRV_REFUSED', ('SK_ID_CURR', 'SK_ID_CURR'),
            PREVIOUS_APPLICATION_AGGREGATION_RECIPIES
        ).aggregate(prv_refused).set_index('SK_ID_CURR')

        feats = pd.concat([feats_all, feats_approved, feats_refused], axis=1)

        return feats.reset_index()


if __name__ == "__main__":
    feat_path = PreviousApplicationAggregation(const.FEATDIR).create_feature(0)

