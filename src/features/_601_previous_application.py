import pandas as pd

import const
from features import FeatherFeatureIDFromDF
from features._000_cleaning import previous_application_cleaning


class PreviousApplication(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_prv

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(prv, random_state: int, args):
        """
        args: list
            number_of_applications

        """
        prv = previous_application_cleaning(prv)

        feats = pd.DataFrame({'SK_ID_CURR': prv['SK_ID_CURR'].unique()})

        prv_sorted = prv.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
        prv_sorted_groupby = prv_sorted.groupby(by=['SK_ID_CURR'])

        # 前回の申し込みが承認されたかどうか
        prv_sorted['WAS_APPROVED'] = (
                prv_sorted['NAME_CONTRACT_STATUS'] == 'Approved'
        ).astype('int')
        g = prv_sorted_groupby['WAS_APPROVED'].last().reset_index()
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # 前回の申し込みが拒否されたかどうか
        prv_sorted['WAS_REFUSED'] = (
                prv_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
        g = prv_sorted_groupby['WAS_REFUSED'].last().reset_index()
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # SK_ID_CURR内のSK_ID_PREVのユニーク数
        g = prv_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'NUM_OF_PRV'}, inplace=True)
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # Refusedされた申込みの割合
        g = prv_sorted.groupby(by=['SK_ID_CURR'])['WAS_REFUSED'].mean().reset_index()
        g.rename(index=str, columns={'WAS_REFUSED': 'RATIO_OF_REFUSED_APPLICATIONS'},
                 inplace=True)
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        # 最後の申し込みがRevolving loanかどうか
        prv_sorted['WAS_REVOLVING_LOAN'] = (
                prv_sorted['NAME_CONTRACT_TYPE'] == 'Revolving loans'
        ).astype('int')
        g = prv_sorted.groupby(by=['SK_ID_CURR'])['WAS_REVOLVING_LOAN'].last().reset_index()
        feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        for number in args:
            prv_tail = prv_sorted_groupby.tail(number)
            tail_groupby = prv_tail.groupby(by=['SK_ID_CURR'])

            # CNT_PAYMENTのある期間平均
            g = tail_groupby['CNT_PAYMENT'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={'CNT_PAYMENT': 'TERM_OF_LAST_{}_CREDITS_MEAN'.format(number)},
                     inplace=True)
            feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

            # n個前の申し込み承認が出た日付の平均
            g = tail_groupby['DAYS_DECISION'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={
                         'DAYS_DECISION': 'DAYS_DECISION_ABOUT_LAST_{}_CREDITS_MEAN'.format(number)
                     },
                     inplace=True)
            feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

            # 最初の引き出し日付の平均
            g = tail_groupby['DAYS_FIRST_DRAWING'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={
                         'DAYS_FIRST_DRAWING': 'DAYS_FIRST_DRAWING_LAST_{}_CREDITS_MEAN'.format(number)
                     },
                     inplace=True)
            feats = feats.merge(g, on=['SK_ID_CURR'], how='left')

        return feats


if __name__ == "__main__":
    import json
    from collections import OrderedDict

    conf = json.load(open('./configs/lightgbm17.json'))
    print()

    PreviousApplication(const.FEATDIR).create_feature(0, *conf['feature']['use_features'][12][1])
