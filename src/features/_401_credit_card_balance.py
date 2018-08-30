import pandas as pd

import const
from features import FeatherFeatureIDFromDF
from features._000_cleaning import credit_card_balance_cleaning


class CreditCardBalance(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_ccb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(ccb: pd.DataFrame, random_state: int):
        ccb = credit_card_balance_cleaning(ccb)

        ccb['number_of_installments'] = ccb.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV']
        )['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()['CNT_INSTALMENT_MATURE_CUM']

        # ccb['CREDIT_CARD_MAX_LOADING_OF_CREDIT_LIMIT'] = ccb.groupby(
        #     by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
        #     lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()
        # ).reset_index()[0]

        feats = pd.DataFrame({'SK_ID_CURR': ccb['SK_ID_CURR'].unique()})

        groupby = ccb.groupby(by=['SK_ID_CURR'])

        # SK_ID_CURR内のSK_ID_PREVの数
        g = groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'NUMBER_OF_LOANS'}, inplace=True)
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # SK_DPDの平均
        g = groupby['SK_DPD'].agg('mean').reset_index()
        g.rename(index=str, columns={'SK_DPD': 'AVERAGE_OF_DAYS_PAST_DUE'},
                 inplace=True)
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # ATMからの引き出し回数
        g = groupby['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'DRAWINGS_ATM'},
                 inplace=True)
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # 引き出し総額
        g = groupby['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'DRAWINGS_TOTAL'},
                 inplace=True)
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['number_of_installments'].agg('sum').reset_index()
        g.rename(index=str, columns={'number_of_installments': 'TOTAL_INSTALLMENTS'},
                 inplace=True)
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # g = groupby['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
        # g.rename(index=str, columns={'credit_card_max_loading_of_credit_limit': 'avg_loading_of_credit_limit'},
        #          inplace=True)
        # feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # キャッシュカード利用率
        feats['CASH_CARD_RATIO'] = (
                feats['DRAWINGS_ATM'] / feats['DRAWINGS_TOTAL']
        )

        # ローンごとの支払い回数平
        feats['INSTALLMENTS_PER_LOAN'] = (
                feats['TOTAL_INSTALLMENTS'] / feats['NUMBER_OF_LOANS']
        )

        # AMT_BALANCEの月別差分平均
        ccb_sorted = ccb.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
        ccb_sorted['BALANCE_MONTHLY_DIFF'] = ccb_sorted.groupby(
            by=['SK_ID_CURR']
        )['AMT_BALANCE'].diff()

        g = ccb_sorted.groupby('SK_ID_CURR')['BALANCE_MONTHLY_DIFF'].agg('mean').reset_index()
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        feats.columns = [
            "CCB_" + _c if _c not in ['SK_ID_CURR'] else _c for _c in feats.columns
        ]

        return feats


if __name__ == '__main__':
    feat_path = CreditCardBalance(const.FEATDIR).create_feature(0)
    print(feat_path)
