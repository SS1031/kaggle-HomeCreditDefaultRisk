import pandas as pd
import numpy as np

from multiprocessing import Pool

import const
from features import FeatherFeatureIDFromDF


pcb = pd.read_feather(const.in_pcb)


def pcb_wavg(x): return np.average(x, weights=-1/pcb.loc[x.index, 'MONTHS_BALANCE'], axis=0)


def wrapper_pcb_wavg(g): return g.apply(pcb_wavg)


class POSCASHBalanceWeightedAvg(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_pcb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(df: pd.DataFrame, random_state: int):

        assert df.equals(pcb)

        def mprocess(grouped, cols, num_processes=4):
            with Pool(num_processes) as pool:
                ret_list = pool.map(wrapper_pcb_wavg, [grouped[col] for col in cols])
            return pd.concat(ret_list, axis=1)

        pcb_wavg_cols = ['CNT_INSTALMENT',
                         'CNT_INSTALMENT_FUTURE',
                         'SK_DPD',
                         'SK_DPD_DEF']

        grouped = df[['SK_ID_CURR'] + pcb_wavg_cols].groupby('SK_ID_CURR')

        ret = mprocess(grouped, pcb_wavg_cols)
        ret.columns = ["PCB_WAVG_" + _c for _c in ret.columns]

        return ret.reset_index()


if __name__ == "__main__":
    pcb_wavg_path = POSCASHBalanceWeightedAvg(const.FEATDIR).create_feature(0)
