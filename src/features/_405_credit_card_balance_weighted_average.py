
import pandas as pd
import numpy as np

from multiprocessing import Pool

import const
from features import FeatherFeatureIDFromDF


# multiprocessingはクラス内のローカル関数だと動作しないのでしょうがなく外部化
# https://stackoverflow.com/questions/8804830/python-multiprocessing-pickling-error
ccb = pd.read_feather(const.in_ccb)


def ccb_wavg(x): return np.average(x, weights=-1/ccb.loc[x.index, 'MONTHS_BALANCE'], axis=0)


def wrapper_ccb_wavg(g): return g.apply(ccb_wavg)


class CreditCardBalanceWeightedAvg(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_ccb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(df: pd.DataFrame, random_state: int):

        def mprocess(grouped, cols, num_processes=4):
            with Pool(num_processes) as pool:
                ret_list = pool.map(wrapper_ccb_wavg, [grouped[col] for col in cols])
            return pd.concat(ret_list, axis=1)

        # CNT系の特徴量のみ重み付き平均を実施
        ccb_wavg_cols = [
            "CNT_DRAWINGS_ATM_CURRENT",
            "CNT_DRAWINGS_CURRENT",
            "CNT_DRAWINGS_OTHER_CURRENT",
            "CNT_DRAWINGS_POS_CURRENT",
            "CNT_INSTALMENT_MATURE_CUM"
        ]

        grouped = df[['SK_ID_CURR'] + ccb_wavg_cols].groupby('SK_ID_CURR')
        ret = mprocess(grouped, ccb_wavg_cols)
        ret.columns = ["CCB_WAVG_" + _c for _c in ret.columns]

        return ret.reset_index()


if __name__ == "__main__":
    ccb_wavg_path = CreditCardBalanceWeightedAvg(const.FEATDIR).create_feature(0)
    # pcb_wavg_path = POSCASHBalanceWeightedAvg(const.FEATDIR).create_feature(0)
