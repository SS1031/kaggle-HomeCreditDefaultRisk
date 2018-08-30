import os
import gc
import time
import pprint
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import const
from features._001_load import datasets
from models.lightgbm import LightGBM
from models.xgboost import XGBoost
from utils import dump_json_log
from utils import update_result_summary
import sys

from collections import OrderedDict

import warnings

warnings.filterwarnings('ignore')

models = {
    'lightgbm': LightGBM,
    'xgboost': XGBoost
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm0.json')
    parser.add_argument('--debug', default='False')

    options = parser.parse_args()

    configs = json.load(open(options.config), object_pairs_hook=OrderedDict)
    # debug = ('debug' in options.config)

    train_results = []
    for iconf, conf in enumerate(configs['config_set'], start=1):
        print("Settings {}: ".format(iconf))
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(conf)
        trn, tst, categorical_features = datasets(conf['feature'],
                                                  random_state=conf['random_seed'],
                                                  debug=options.debug)
        print("###################", trn.shape, tst.shape)

        y = trn[const.TARGET_COL]
        trn.drop(columns=[const.TARGET_COL], inplace=True)
        gc.collect()
        feature_cols = [_c for _c in trn.columns if _c not in ['SK_ID_CURR']]

        folds_preds = pd.DataFrame([0] * len(y), columns=['preds'], index=y.index)
        sub_preds = pd.DataFrame(index=tst.index)
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=conf['random_seed'])

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trn, y), 1):
            model = models[conf['model']['name']]()
            start_time = time.time()

            trn_x, trn_y = trn[feature_cols].iloc[trn_idx], y.iloc[trn_idx]
            val_x, val_y = trn[feature_cols].iloc[val_idx], y.iloc[val_idx]

            booster, result = model.train_and_predict(
                trn_x=trn_x,
                trn_y=trn_y,
                val_x=val_x,
                val_y=val_y,
                params=conf['model'],
            )

            # best_iteration = booster.best_iteration
            # folds_preds.loc[val_idx, 'preds'] += (
            #     booster.predict(val_x, num_iteration=best_iteration) / len(configs['config_set'])
            # )
            #
            # sub_preds['preds{}_{}'.format(iconf, n_fold)] = booster.predict(
            #     tst[feature_cols],
            #     num_iteration=best_iteration
            # )

            folds_preds.loc[val_idx, 'preds'] += (
                model.predict(booster, val_x) / len(configs['config_set'])
            )
            sub_preds['preds{}_{}'.format(iconf, n_fold)] = (
                model.predict(booster, tst[feature_cols])
            )

            print('Fold {}_{} AUC : {}'.format(
                iconf, n_fold, roc_auc_score(val_y, model.predict(booster, val_x))
            ))

            train_time = time.time() - start_time
            train_results = model.append_result(train_results,
                                                result,
                                                iconf,
                                                n_fold,
                                                booster,
                                                train_time)

    print('Full AUC score %.6f' % roc_auc_score(y, folds_preds['preds']))
    dump_json_log(options, train_results, const.OUTDIR)
    update_result_summary(os.path.basename(__file__), options, train_results)
    sbmt = pd.concat([tst.SK_ID_CURR, sub_preds.mean(axis=1)], axis=1)
    sbmt.columns = ['SK_ID_CURR', 'TARGET']
    sbmt.to_csv(os.path.join(const.OUTDIR,
                             'sbmt_{}.csv'.format(
                                 os.path.splitext(os.path.basename(options.config))[0]
                             )),
                index=False)


if __name__ == "__main__":
    main()
