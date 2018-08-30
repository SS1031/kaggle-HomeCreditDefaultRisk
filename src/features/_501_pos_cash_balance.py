import sys

import pandas as pd

import const
from features import FeatherFeatureIDFromDF
from functools import partial

from features._999_utils import parallel_apply
from features._999_utils import add_features_in_group
from features._999_utils import add_trend_feature


class POSCASHBalance(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_pcb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(pcb: pd.DataFrame, random_state, *args):
        """
        args[0]: list
            last_k_agg_periods
        args[1]: list
            last_k_trend_periods
        args[2]: int
            num_workers

        """

        pcb['is_contract_status_completed'] = pcb['NAME_CONTRACT_STATUS'] == 'Completed'
        pcb['pos_cash_paid_late'] = (pcb['SK_DPD'] > 0).astype(int)
        pcb['pos_cash_paid_late_with_tolerance'] = (pcb['SK_DPD_DEF'] > 0).astype(int)

        feats = pd.DataFrame({'SK_ID_CURR': pcb['SK_ID_CURR'].unique()})

        groupby = pcb.groupby(['SK_ID_CURR'])

        func = partial(POSCASHBalance.generate_features,
                       agg_periods=args[0],
                       trend_periods=args[1])

        g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                           num_workers=args[2]).reset_index()

        feats = feats.merge(g, on='SK_ID_CURR', how='left')
        feats.columns = [
            "PCB_" + _c if _c not in ['SK_ID_CURR'] else _c for _c in feats.columns
        ]

        return feats

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):
        _one_time = POSCASHBalance.one_time_features(gr)
        _all = POSCASHBalance.all_installment_features(gr)
        _agg = POSCASHBalance.last_k_installment_features(gr, agg_periods)
        _trend = POSCASHBalance.trend_in_last_k_installment_features(gr, trend_periods)
        _last = POSCASHBalance.last_loan_features(gr)

        features = {**_one_time, **_all, **_agg, **_trend, **_last}

        return pd.Series(features)

    @staticmethod
    def one_time_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], inplace=True)

        features = {}
        features['pos_cash_remaining_installments'] = gr_['CNT_INSTALMENT_FUTURE'].tail(1).values[0]
        features['pos_cash_completed_contracts'] = gr_['is_contract_status_completed'].agg('sum')

        return features

    @staticmethod
    def all_installment_features(gr):
        return POSCASHBalance.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_.iloc[:period]

            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_.iloc[:period]

            features = add_trend_feature(
                features, gr_period, 'SK_DPD', '{}_period_trend_'.format(period)
            )

            features = add_trend_feature(
                features, gr_period, 'SK_DPD_DEF', '{}_period_trend_'.format(period)
            )

            features = add_trend_feature(
                features, gr_period, 'CNT_INSTALMENT_FUTURE', '{}_period_trend_'.format(period)
            )

        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
        last_installment_id = gr_['SK_ID_PREV'].iloc[0]
        gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

        features = {}
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                         ['count', 'sum', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                         ['mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')

        return features


if __name__ == "__main__":
    import json
    from collections import OrderedDict

    conf = json.load(open('./configs/lightgbm25.json'), object_pairs_hook=OrderedDict)
    pcb_path = POSCASHBalance(const.FEATDIR).create_feature(
        None,
        *conf['feature']['use_features'][18][1]
    )
