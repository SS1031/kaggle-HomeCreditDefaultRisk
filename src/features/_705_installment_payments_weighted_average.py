import pandas as pd
import numpy as np

from multiprocessing import Pool

import const
from features import FeatherFeatureIDFromDF


# multiprocessingはクラス内のローカル関数だと動作しないのでしょうがなく外部化
# https://stackoverflow.com/questions/8804830/python-multiprocessing-pickling-error
inp = pd.read_feather(const.in_inp)


def inp_wavg(x): return np.average(x, weights=-1/inp.loc[x.index, 'DAYS_INSTALMENT'], axis=0)


def wrapper_inp_wavg(g): return g.apply(inp_wavg)


class InstallmentPaymentsWeightedAverage(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_inp

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(inp: pd.DataFrame, random_state: int):

        inp['PAID_LATE_IN_DAYS'] = inp['DAYS_ENTRY_PAYMENT'] - inp['DAYS_INSTALMENT']
        inp['PAID_LATE'] = (inp['PAID_LATE_IN_DAYS'] > 0).astype(int)
        inp['PAID_OVER_AMOUNT'] = inp['AMT_PAYMENT'] - inp['AMT_INSTALMENT']
        inp['PAID_OVER'] = (inp['PAID_OVER_AMOUNT'] > 0).astype(int)

        def mprocess(grouped, cols, num_processes=4):
            with Pool(num_processes) as pool:
                ret_list = pool.map(wrapper_inp_wavg, [grouped[col] for col in cols])
            return pd.concat(ret_list, axis=1)

        inp_wavg_cols = [
            "PAID_LATE_IN_DAYS",
            "PAID_LATE",
            "PAID_OVER_AMOUNT",
            "PAID_OVER",
            "AMT_PAYMENT",
        ]

        grouped = inp[['SK_ID_CURR'] + inp_wavg_cols].groupby('SK_ID_CURR')
        ret = mprocess(grouped, inp_wavg_cols)
        ret.columns = ["INP_WAVG_" + _c for _c in ret.columns]

        return ret.reset_index()


if __name__ == "__main__":
    inp_wavg_path = InstallmentPaymentsWeightedAverage(const.FEATDIR).create_feature(0)
    inp_wavg = pd.read_feather(inp_wavg_path)
