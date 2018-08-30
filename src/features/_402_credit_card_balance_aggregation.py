import pandas as pd

import const

from features import FeatherFeatureIDFromDF
from features._000_cleaning import credit_card_balance_cleaning
from features._999_utils import GroupbyAggregate


class CreditCardBalanceAggregation(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_ccb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(ccb: pd.DataFrame, random_state=None):
        ccb = credit_card_balance_cleaning(ccb)

        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['AMT_BALANCE',
                           'AMT_CREDIT_LIMIT_ACTUAL',
                           'AMT_DRAWINGS_ATM_CURRENT',
                           'AMT_DRAWINGS_CURRENT',
                           'AMT_DRAWINGS_OTHER_CURRENT',
                           'AMT_DRAWINGS_POS_CURRENT',
                           'AMT_PAYMENT_CURRENT',
                           'CNT_DRAWINGS_ATM_CURRENT',
                           'CNT_DRAWINGS_CURRENT',
                           'CNT_DRAWINGS_OTHER_CURRENT',
                           'CNT_INSTALMENT_MATURE_CUM',
                           'MONTHS_BALANCE',
                           'SK_DPD',
                           'SK_DPD_DEF']:
                CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]

        feats = GroupbyAggregate('CCB', ('SK_ID_CURR', 'SK_ID_CURR'),
                                 CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES).aggregate(ccb)

        return feats


if __name__ == "__main__":
    feat_path = CreditCardBalanceAggregation(const.FEATDIR).create_feature(0)
