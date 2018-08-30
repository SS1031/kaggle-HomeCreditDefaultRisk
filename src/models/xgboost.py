import copy
from typing import List, Tuple

import pandas as pd
import xgboost as xgb

from models.base import Model


class XGBoost(Model):

    def train_and_predict(self, trn_x, trn_y, val_x, val_y, params):

        if type(trn_x) != pd.DataFrame or type(val_x) != pd.DataFrame:
            raise ValueError('Parameter train and valid must be pandas.DataFrame')

        if list(trn_x.columns) != list(val_x.columns):
            raise ValueError('Train and valid must have a same column list')

        d_train = xgb.DMatrix(trn_x, label=trn_y.values)
        d_valid = xgb.DMatrix(val_x, label=val_y.values)

        eval_results = {}
        booster = xgb.train(params['model_params'],
                            d_train,
                            evals=[(d_train, 'train'), (d_valid, 'valid')],
                            evals_result=eval_results,
                            verbose_eval=100,
                            **params['train_params'])

        return booster, eval_results

    def predict(self, booster, data):

        d_test = xgb.DMatrix(data)

        return booster.predict(d_test)

    def append_result(self, train_results, result, iconf, n_fold, booster, train_time):

        best_iteration = booster.best_iteration
        train_results.append({
            'n_fold': "{}_{}".format(iconf, n_fold),
            'train_auc': result['train']['auc'][best_iteration],
            'valid_auc': result['valid']['auc'][best_iteration],
            'best_iteration': best_iteration,
            'train_time': train_time,
            'feature_importance': booster.get_score()
        })

        return train_results
