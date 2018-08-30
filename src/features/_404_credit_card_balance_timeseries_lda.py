import gc
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from features._000_cleaning import credit_card_balance_cleaning
from features._999_utils import monthly_balance_timeseries_table

import const
from features import FeatherFeatureIDFromDF


class CreditCardBalanceTimeseriesLDA(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_ccb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(ccb: pd.DataFrame, random_state: int, *args):
        """
        args[0]: list
            last_k_months
        args[1]: list
            ncomps
        """

        ccb_ts_lda = pd.DataFrame()
        for period, ncomp in zip(args[0], args[1]):
            ccb_timeseries = monthly_balance_timeseries_table(ccb.copy(), period=period)
            print("TimeSeries Data Shape:", ccb_timeseries.shape)
            ts_feat_cols = [c for c in ccb_timeseries.columns if c not in ['SK_ID_CURR']]

            print("Period{}, LDA Process ...".format(period))
            lda = LatentDirichletAllocation(n_components=ncomp,
                                            random_state=random_state,
                                            n_jobs=-1)
            numpy_ccb_ts_lda = lda.fit_transform(ccb_timeseries[ts_feat_cols])
            part_ccb_ts_lda = pd.DataFrame(
                numpy_ccb_ts_lda,
                columns=["CCB_PERIOD{}_TS_LDA{}".format(period, i) for i in range(ncomp)],
                index=ccb_timeseries.SK_ID_CURR
            )
            ccb_ts_lda = pd.concat([ccb_ts_lda, part_ccb_ts_lda], axis=1)

        return ccb_ts_lda.reset_index()


if __name__ == "__main__":
    import json
    from collections import OrderedDict
    configs = json.load(open('./configs/lightgbm27.json'), object_pairs_hook=OrderedDict)
    feat_path = CreditCardBalanceTimeseriesLDA(const.FEATDIR).create_feature(
        configs['config_set'][0]['random_seed'],
        *configs['config_set'][0]['feature']['use_features'][16][1]
    )
