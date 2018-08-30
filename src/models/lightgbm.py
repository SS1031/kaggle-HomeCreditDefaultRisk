import copy
from typing import List, Tuple

import lightgbm as lgb
import pandas as pd
from lightgbm import Booster

from models.base import Model


class LightGBM(Model):

    def train_and_predict(self, trn_x, trn_y, val_x, val_y, params) -> Tuple[Booster, dict]:

        if type(trn_x) != pd.DataFrame or type(val_x) != pd.DataFrame:
            raise ValueError('Parameter train and valid must be pandas.DataFrame')

        if list(trn_x.columns) != list(val_x.columns):
            raise ValueError('Train and valid must have a same column list')

        # predictors = trn_x.columns.drop(target)

        d_train = lgb.Dataset(trn_x, label=trn_y.values)
        d_valid = lgb.Dataset(val_x, label=val_y.values)

        eval_results = {}
        model = lgb.train(params['model_params'],
                          d_train,
                          valid_sets=[d_train, d_valid],
                          valid_names=['train', 'valid'],
                          evals_result=eval_results,
                          verbose_eval=100,
                          **params['train_params'])

        return model, eval_results

    def predict(self, booster, data):
        best_iteration = booster.best_iteration
        return booster.predict(data, num_iteration=best_iteration)

    def append_result(self, train_results, result, iconf, n_fold, booster, train_time):

        best_iteration = booster.best_iteration
        train_results.append({
            'n_fold': "{}_{}".format(iconf, n_fold),
            'train_auc': result['train']['auc'][best_iteration],
            'valid_auc': result['valid']['auc'][best_iteration],
            'best_iteration': best_iteration,
            'train_time': train_time,
            'feature_importance': {name: int(score) for name, score in
                                   zip(booster.feature_name(),
                                       booster.feature_importance())}
        })

        return train_results
