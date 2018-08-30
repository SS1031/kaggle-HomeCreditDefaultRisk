import pandas as pd

import const

from features import FeatherFeatureIDFromDF
from features._999_utils import GroupbyAggregate


class InstallmentPaymentsAggregation(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_inp

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(inp: pd.DataFrame, random_state: int):

        INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['AMT_INSTALMENT',
                           'AMT_PAYMENT',
                           'DAYS_ENTRY_PAYMENT',
                           'DAYS_INSTALMENT',
                           'NUM_INSTALMENT_NUMBER',
                           'NUM_INSTALMENT_VERSION']:
                INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))

        INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'],
                                                       INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]

        feats = GroupbyAggregate('INP', ('SK_ID_CURR', 'SK_ID_CURR'),
                                 INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES).aggregate(inp)

        return feats


if __name__ == "__main__":

    feat_path = InstallmentPaymentsAggregation(const.FEATDIR).create_feature(0)
