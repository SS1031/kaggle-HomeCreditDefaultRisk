import gc
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

import const
from features import FeatherFeatureIDFromDF
from features._999_utils import monthly_balance_timeseries_table


class POSCASHBalanceTimeseriesLDA(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_pcb

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(pcb: pd.DataFrame, random_state: int, *args):
        """
        args[0]: list
            last_k_months
        args[1]: list
            ncomps
        """

        pcb_ts_lda = pd.DataFrame()

        for period, ncomp in zip(args[0], args[1]):
            pcb_timeseries = monthly_balance_timeseries_table(pcb.copy(), period=period)
            print("TimeSeries Data Shape:", pcb_timeseries.shape)
            ts_feat_cols = [c for c in pcb_timeseries.columns if c not in ['SK_ID_CURR']]

            print("Period{}, LDA Process ...".format(period))
            lda = LatentDirichletAllocation(n_components=ncomp,
                                            random_state=random_state,
                                            n_jobs=-1)
            numpy_pcb_ts_lda = lda.fit_transform(pcb_timeseries[ts_feat_cols])

            part_ccb_ts_lda = pd.DataFrame(
                numpy_pcb_ts_lda,
                columns=["PCB_PERIOD{}_TS_LDA{}".format(period, i) for i in range(ncomp)],
                index=pcb_timeseries.SK_ID_CURR
            )
            pcb_ts_lda = pd.concat([pcb_ts_lda, part_ccb_ts_lda], axis=1)

        print("Done.")
        return pcb_ts_lda.reset_index()


if __name__ == "__main__":
    # feat_path = POSCASHBalanceTimeseriesLDA(const.FEATDIR).create_feature(0)
    import json
    from collections import OrderedDict
    configs = json.load(open('./configs/lightgbm27.json'), object_pairs_hook=OrderedDict)
    feat_path = POSCASHBalanceTimeseriesLDA(const.FEATDIR).create_feature(
        configs['config_set'][0]['random_seed'],
        *configs['config_set'][0]['feature']['use_features'][18][1]
    )
