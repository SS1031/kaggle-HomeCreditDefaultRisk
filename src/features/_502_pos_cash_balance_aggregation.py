import pandas as pd


import const

from features import FeatherFeatureIDFromDF
from features._999_utils import GroupbyAggregate


class POSCASHBalanceAggregation(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_pcb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(pcb: pd.DataFrame, random_state: int):

        POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['MONTHS_BALANCE',
                           'SK_DPD',
                           'SK_DPD_DEF'
                           ]:
                POS_CASH_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
        POS_CASH_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_BALANCE_AGGREGATION_RECIPIES)]

        feats = GroupbyAggregate('PCB', ('SK_ID_CURR', 'SK_ID_CURR'),
                                 POS_CASH_BALANCE_AGGREGATION_RECIPIES).aggregate(pcb)

        return feats


if __name__ == "__main__":
    feat_path = POSCASHBalanceAggregation(const.FEATDIR).create_feature(0)
