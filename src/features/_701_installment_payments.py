import pandas as pd

from features import FeatherFeatureIDFromDF
from functools import partial

from features._999_utils import parallel_apply
from features._999_utils import add_features_in_group
from features._999_utils import add_trend_feature
from features._999_utils import add_last_k_features_fractions

import const


class InstallmentPayments(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_inp

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(inp, random_state: int, *args):
        """
        args[0]: list
            last_k_agg_periods
        args[1]: list
            last_k_agg_period_fractions
        args[2]:
            last_k_trend_periods
        args[3]:
            num_workers
        """

        inp['PAID_LATE_IN_DAYS'] = inp['DAYS_ENTRY_PAYMENT'] - inp['DAYS_INSTALMENT']
        inp['PAID_LATE'] = (inp['PAID_LATE_IN_DAYS'] > 0).astype(int)
        inp['PAID_OVER_AMOUNT'] = inp['AMT_PAYMENT'] - inp['AMT_INSTALMENT']
        inp['PAID_OVER'] = (inp['PAID_OVER_AMOUNT'] > 0).astype(int)

        feats = pd.DataFrame({'SK_ID_CURR': inp['SK_ID_CURR'].unique()})
        groupby = inp.groupby(['SK_ID_CURR'])

        func = partial(InstallmentPayments.generate_features,
                       agg_periods=args[0],
                       trend_periods=args[2])

        g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                           num_workers=args[3]).reset_index()
        feats = feats.merge(g, on='SK_ID_CURR', how='left')

        g = add_last_k_features_fractions(feats, 'SK_ID_CURR',
                                          period_fractions=args[1])
        feats = feats.merge(g, on='SK_ID_CURR', how='left')

        feats.columns = [
            "INP_" + _c if _c not in ['SK_ID_CURR'] else _c for _c in feats.columns
        ]

        return feats

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):

        _all = InstallmentPayments.all_installment_features(gr)
        _agg = InstallmentPayments.last_k_installment_features(gr, agg_periods)
        _trend = InstallmentPayments.trend_in_last_k_installment_features(gr, trend_periods)
        _last = InstallmentPayments.last_loan_features(gr)

        features = {**_all, **_agg, **_trend, **_last}

        return pd.Series(features)

    @staticmethod
    def all_installment_features(gr):
        return InstallmentPayments.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'ALL_INSTALLMENT_'
                gr_period = gr_.copy()
            else:
                period_name = 'LAST_{}_'.format(period)
                gr_period = gr_[gr_['DAYS_INSTALMENT'] > (-1) * period]

            features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'iqr'],
                                             period_name)

            features = add_features_in_group(features, gr_period, 'PAID_LATE_IN_DAYS',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'kurt', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'PAID_LATE',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'PAID_OVER_AMOUNT',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'kurt'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'PAID_OVER',
                                             ['count', 'mean'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_[gr_['DAYS_INSTALMENT'] > (-1) * period]

            features = add_trend_feature(features, gr_period,
                                         'PAID_LATE_IN_DAYS', '{}_PERIOD_TREND_'.format(period))
            features = add_trend_feature(features, gr_period,
                                         'PAID_OVER_AMOUNT', '{}_PERIOD_TREND_'.format(period))
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        last_inp_ids = gr_[gr_['DAYS_INSTALMENT'] == gr_['DAYS_INSTALMENT'].max()]['SK_ID_PREV']
        gr_ = gr_[gr_['SK_ID_PREV'].isin(last_inp_ids)]

        features = {}
        features = add_features_in_group(features, gr_,
                                         'PAID_LATE_IN_DAYS',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')

        features = add_features_in_group(features, gr_,
                                         'PAID_LATE',
                                         ['count', 'mean'],
                                         'last_loan_')

        features = add_features_in_group(features, gr_,
                                         'PAID_OVER_AMOUNT',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')

        features = add_features_in_group(features, gr_,
                                         'PAID_OVER',
                                         ['count', 'mean'],
                                         'last_loan_')
        return features


if __name__ == '__main__':
    import json
    from collections import OrderedDict

    conf = json.load(open('./configs/lightgbm17.json'))
    inp_path = InstallmentPayments(const.FEATDIR).create_feature(
        0, *conf['feature']['use_features'][14][1]
    )
