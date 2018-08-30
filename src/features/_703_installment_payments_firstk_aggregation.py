import pandas as pd
from scipy.stats import kurtosis
from tqdm import tqdm

import const
from features import FeatherFeatureIDFromDF


class InstallmentPaymentsFirstKAggregation(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_inp

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(inp: pd.DataFrame, random_state: int, *args):
        """
        args[0] : list
            aggregation periods

        """

        inp['PAID_LATE_IN_DAYS'] = inp['DAYS_ENTRY_PAYMENT'] - inp['DAYS_INSTALMENT']
        inp['PAID_LATE'] = (inp['PAID_LATE_IN_DAYS'] > 0).astype(int)
        inp['PAID_OVER_AMOUNT'] = inp['AMT_PAYMENT'] - inp['AMT_INSTALMENT']
        inp['PAID_OVER'] = (inp['PAID_OVER_AMOUNT'] > 0).astype(int)

        inp.sort_values(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER'], inplace=True)
        aggregations = pd.DataFrame(index=inp.SK_ID_CURR.unique())
        aggregations.index.name = 'SK_ID_CURR'

        for period in tqdm(args[0]):
            col_prefix = 'INP_FIRST{}_'.format(period)

            grp = inp[inp.NUM_INSTALMENT_NUMBER <= period].copy().groupby(['SK_ID_CURR'])
            period_agg = grp.agg({
                'NUM_INSTALMENT_VERSION': [pd.Series.nunique, 'mean', 'std', 'max', 'median'],
                'PAID_LATE_IN_DAYS': ['sum', 'mean', 'max', 'min', 'std', 'median', kurtosis],
                'PAID_LATE': ['sum', 'mean'],
                'PAID_OVER_AMOUNT': ['sum', 'mean', 'max', 'min', 'std'],
                'PAID_OVER': ['sum', 'mean']
            })
            period_agg.columns = [
                col_prefix + col[0] + '_' + col[1].upper() for col in period_agg.columns.values
            ]
            aggregations = pd.concat([aggregations, period_agg], axis=1)

        return aggregations.reset_index()


if __name__ == "__main__":
    import json
    from collections import OrderedDict
    configs = json.load(open('./configs/lightgbm29.json'), object_pairs_hook=OrderedDict)
    feat_path = InstallmentPaymentsFirstKAggregation(const.FEATDIR).create_feature(
        configs['config_set'][0]['random_seed'],
        *configs['config_set'][0]['feature']['use_features'][21][1]
    )

    # inp = pd.read_feather(const.in_inp)

    # inp['PAID_LATE_IN_DAYS'] = inp['DAYS_ENTRY_PAYMENT'] - inp['DAYS_INSTALMENT']
    # inp['PAID_LATE'] = (inp['PAID_LATE_IN_DAYS'] > 0).astype(int)
    # inp['PAID_OVER_AMOUNT'] = inp['AMT_PAYMENT'] - inp['AMT_INSTALMENT']
    # inp['PAID_OVER'] = (inp['PAID_OVER_AMOUNT'] > 0).astype(int)

    # inp.sort_values(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER'], inplace=True)
    # aggregations = pd.DataFrame(index=inp.SK_ID_CURR.unique())
    # aggregations.index.name = 'SK_ID_CURR'

    # for period in tqdm([2]):
    #     col_prefix = 'INP_FIRST{}_'.format(period)

    #     grp = inp[inp.NUM_INSTALMENT_NUMBER <= period].copy().groupby(['SK_ID_CURR'])
    #     period_agg = grp.agg({
    #         'NUM_INSTALMENT_VERSION': [pd.Series.nunique, 'mean', 'std', 'max', 'median'],
    #         'PAID_LATE_IN_DAYS': ['sum', 'mean', 'max', 'min', 'std', 'median', kurtosis],
    #         'PAID_LATE': ['sum', 'mean'],
    #         'PAID_OVER_AMOUNT': ['sum', 'mean', 'max', 'min', 'std'],
    #         'PAID_OVER': ['sum', 'mean']
    #     })
    #     period_agg.columns = [
    #         col_prefix + col[0] + '_' + col[1].upper() for col in period_agg.columns.values
    #     ]
    #     print(aggregations.index.name)
    #     aggregations = pd.concat([aggregations, period_agg], axis=1)

