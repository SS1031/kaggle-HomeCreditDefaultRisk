import pandas as pd

import const

from features import FeatherFeatureIDFromDF
from features._000_cleaning import credit_card_balance_cleaning


def cnt_up(grp):
    return (grp.diff() > 0).sum()


def cnt_down(grp):
    return (grp.diff() < 0).sum()


def no_change(grp):
    return (grp.diff() == 0).sum()


class CreditCardBalanceUpDown(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_ccb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(ccb: pd.DataFrame, random_state: int):
        ccb = credit_card_balance_cleaning(ccb)

        ccb = ccb.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE']).reset_index(drop=True)

        ccb_prv_agg = ccb.groupby('SK_ID_PREV').agg({
            'SK_ID_CURR': [pd.Series.unique, 'size'],
            'AMT_BALANCE': ['max', cnt_up, cnt_down, no_change]
        })

        # カラム名の変更
        ccb_prv_agg.columns = ['_'.join(col.upper()).strip() for col in ccb_prv_agg.columns.values]

        # SK_ID_CURRのrename
        ccb_prv_agg.rename(columns={'SK_ID_CURR_UNIQUE': 'SK_ID_CURR'}, inplace=True)
        ccb_prv_agg.reset_index(inplace=True)

        # AMT_BALANCEの増加回数、減少回数、変わらなかった回数をすべてのMonthの数で割る
        ccb_log_num = ccb_prv_agg.groupby('SK_ID_CURR')['SK_ID_CURR_size'].sum()
        ccb_updown = ccb_prv_agg.groupby('SK_ID_CURR')[[
            'AMT_BALANCE_CNT_UP',
            'AMT_BALANCE_CNT_DOWN',
            'AMT_BALANCE_NO_CHANGE',
        ]].mean()
        norm_ccb_updown = pd.DataFrame()
        for col in ccb_updown.columns:
            norm_ccb_updown[col] = norm_ccb_updown[col] / (ccb_log_num - 1)

        norm_ccb_updown.columns = ["CCB_" + col for col in norm_ccb_updown.columns]
        norm_ccb_updown.reset_index(inplace=True)

        return norm_ccb_updown


if __name__ == '__main__':
    CreditCardBalanceUpDown(const.FEATDIR).create_feature(0)

