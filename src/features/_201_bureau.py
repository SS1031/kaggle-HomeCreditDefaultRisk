import pandas as pd
import numpy as np

import const
from features import FeatherFeatureIDFromDF
from features._000_cleaning import bureau_cleaning


class Bureau(FeatherFeatureIDFromDF):

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
        features = pd.DataFrame({'SK_ID_CURR': bur['SK_ID_CURR'].unique()})

        bur['BUR_CREDIT_ACTIVE_BINARY'] = (bur.CREDIT_ACTIVE == 'Active').astype(int)
        bur['BUR_CREDIT_ACTIVE*DAYS_CREDIT'] = (bur.BUR_CREDIT_ACTIVE_BINARY * bur.DAYS_CREDIT)
        bur['BUR_CREDIT_ACTIVE*DAYS_CREDIT_ENDDATE'] = (
            bur.BUR_CREDIT_ACTIVE_BINARY * bur.DAYS_CREDIT_ENDDATE)

        # クレジット状態
        bur_credit_active_dummie = pd.get_dummies(bur.CREDIT_ACTIVE)
        bur_credit_active_dummie['SK_ID_CURR'] = bur.SK_ID_CURR
        bur_credit_active = bur_credit_active_dummie.groupby('SK_ID_CURR').agg(['mean', 'sum'])
        bur_credit_active.columns = \
            ["BUR_CREDIT_STATUS_" + c[0].replace(" ", "") + c[1].upper()
             for c in bur_credit_active.columns]
        features = features.merge(bur_credit_active, on=['SK_ID_CURR'], how='left')

        # 以下, groupbyのアグリゲーション
        groupby = bur.groupby(by=['SK_ID_CURR'])

        # bureau内のSK_ID_CURRカウント > 過去bureauの数
        g = groupby['DAYS_CREDIT'].agg('count').reset_index()
        g.rename(index=str, columns={'DAYS_CREDIT': 'BUR_NUMBER_OF_PAST_LOANS'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # log((クレジット適用の経過日数) * (アクティブ or not アクティブ))
        g = groupby['BUR_CREDIT_ACTIVE*DAYS_CREDIT'].agg(['mean', 'max'])
        for c in g.columns:
            g[c] = np.log(g[c] + 0.001)
        g.columns = ['BUR_log({}_BUR_CREDIT_ACTIVE*DAYS_CREDIT)'.format(c.upper())
                     for c in g.columns]
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # 延滞日数のlog
        g = groupby['CREDIT_DAY_OVERDUE'].agg(['mean', 'max'])
        for c in g.columns:
            g[c] = np.log(g[c] + 0.001)
        g.columns = ['BUR_log({}_CREDIT_DAY_OVERDUE)'.format(c.upper()) for c in g.columns]
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # クレジットの残り支払い期間
        g = groupby['DAYS_CREDIT_ENDDATE'].agg(['mean', 'max'])
        for c in g.columns:
            g[c] = np.log(g[c] + 0.001)
        g.columns = ['BUR_log({}_DAYS_CREDIT_ENDDATE)'.format(c.upper()) for c in g.columns]
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # log((クレジットの残り支払い期間) * (アクティブ or not アクティブ))
        g = groupby['BUR_CREDIT_ACTIVE*DAYS_CREDIT_ENDDATE'].agg(['mean', 'max'])
        for c in g.columns:
            g[c] = np.log(g[c] + 0.001)
        g.columns = ['BUR_log({}_BUR_CREDIT_ACTIVE*DAYS_CREDIT_ENDDATE)'.format(c.upper())
                     for c in g.columns]
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # クレジットタイプのユニーク数
        g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()
        g.rename(index=str, columns={'CREDIT_TYPE': 'BUR_NUMBER_OF_LOAN_TYPES'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # log(借入金額総額)
        g = np.log(groupby['AMT_CREDIT_SUM_DEBT'].agg('sum')).reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'BUR_log(sum(AMT_CREDIT_SUM_DEBT))'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # log(与信額総額)
        g = np.log(groupby['AMT_CREDIT_SUM'].agg('sum') + 1).reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM': 'BUR_log(sum(AMT_CREDIT_SUM))'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # クレジットの上限額
        g = np.log(groupby['AMT_CREDIT_SUM_LIMIT'].agg('sum') + 1).reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM': 'BUR_log(sum(AMT_CREDIT_SUM_LIMIT))'},
                 inplace=True)

        # log(延滞額総額)
        g = np.log(groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum') + 1).reset_index()
        g.rename(index=str,
                 columns={'AMT_CREDIT_SUM_OVERDUE': 'BUR_log(sum(AMT_CREDIT_SUM_OVERDUE))'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        # loanタイプ別のbureauレコード割合
        features['BUR_AVERAGE_OF_PAST_LOANS_PER_TYPE'] = \
            features['BUR_NUMBER_OF_PAST_LOANS'] / features['BUREAU_NUMBER_OF_LOAN_TYPES']

        # 借入総額 / 与信総額
        features['BUR_DEBT_CREDIT_RATIO'] = \
            features['BUR_TOTAL_CUSTOMER_DEBT'] / features['BUR_TOTAL_CUSTOMER_CREDIT']

        # 延滞総額 / 借入総額
        features['BUR_OVERDUE_DEBT_RATIO'] = \
            features['BUR_TOTAL_CUSTOMER_OVERDUE'] / features['BUR_TOTAL_CUSTOMER_DEBT']

        return features


if __name__ == "__main__":
    bur_feats = Bureau(const.FEATDIR).create_feature(0)
