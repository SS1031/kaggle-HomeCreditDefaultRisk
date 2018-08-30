import os
import sys
import argparse
import json
import pickle

import numpy as np
import lightgbm as lgb

from hyperopt import hp, tpe, Trials
from hyperopt.fmin import fmin
from hyperopt.pyll.base import scope
from hyperopt import space_eval

import const
from features._001_load import datasets
from utils import dump_json_opt_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/opt/lightgbm0.json')
    parser.add_argument('--debug', default='False')

    # parser.add_argument('--train_only', default=False, action='store_true')
    options = parser.parse_args()
    conf = json.load(open(options.config))

    trn, tst, categorical_features = datasets(conf['feature'],
                                              random_state=conf['random_seed'],
                                              debug=options.debug)

    y = trn[const.TARGET_COL]
    trn.drop(columns=[const.TARGET_COL], inplace=True)

    trn_dataset = lgb.Dataset(data=trn, label=y,
                              categorical_feature=categorical_features,
                              free_raw_data=False)


    def objective(params):
        params['objective'] = "binary"
        params['metric'] = "auc"
        params['boosting_type'] = 'gbdt'
        # params['learning_rate'] = 0.1

        if "win32" == sys.platform:
            params['device'] = "cpu"
        else:
            params['device'] = "gpu"

        cv_results = lgb.cv(train_set=trn_dataset,
                            params=params,
                            nfold=5,
                            stratified=True,
                            num_boost_round=10000,
                            early_stopping_rounds=100,
                            verbose_eval=100,
                            metrics=['auc'])

        return 1 - np.max(cv_results['auc-mean'])


    space = {
        'learning_rate': hp.loguniform('learning_rate', -4.0, -2.3),
        'num_leaves': scope.int(hp.quniform('num_leaves', 20, 100, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 7, 30, 1)),
        'max_bin': scope.int(hp.quniform('max_bin', 180, 500, 2)),
        'min_data_in_leaf': scope.int(hp.quniform('mig_data_in_leaf', 1, 50, 1)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'subsample_freq': scope.int(hp.quniform('subsample_freq', 1, 5, 1)),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.01, 1.0),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0.001, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 0.1),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 0.1),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 0.9, 1.0),
    }

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=conf['max_eval'],
                trials=trials)

    # Get the values of the optimal parameters
    best_params = space_eval(space, best)

    output_path = os.path.basename(options.config).split('.')[0]
    json_output_path = "../data/output/opt/" + output_path + ".bst_params.json"
    json.dump(best_params, open(json_output_path, 'w'), indent=2)

    trial_output_path = "../data/output/opt/" + output_path + ".trials.pkl"

    with open(trial_output_path, 'wb') as out:
        pickle.dump(trials, out)
