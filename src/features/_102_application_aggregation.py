import gc
import pandas as pd

import const

from features import FeatherFeatureTrnTstDF
from features._000_cleaning import application_cleaning
from features._999_utils import GroupbyAggregateDiffs


class ApplicationAggregationAndDiff(FeatherFeatureTrnTstDF):

    @staticmethod
    def input():
        return const.in_app_trn, const.in_app_tst

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(trn: pd.DataFrame, tst: pd.DataFrame, random_state=None):
        trn.drop(columns=['TARGET'], inplace=True)

        trn['is_tst'] = False
        tst['is_tst'] = True
        trn_tst = pd.concat([trn, tst], axis=0).reset_index(drop=True)

        trn_tst = application_cleaning(trn_tst)

        cols_to_agg = [
            'AMT_CREDIT',
            'AMT_ANNUITY',
            'AMT_INCOME_TOTAL',
            'AMT_GOODS_PRICE',
            'EXT_SOURCE_1',
            'EXT_SOURCE_2',
            'EXT_SOURCE_3',
            'OWN_CAR_AGE',
            'REGION_POPULATION_RELATIVE',
            'DAYS_REGISTRATION',
            'CNT_CHILDREN',
            'CNT_FAM_MEMBERS',
            'DAYS_ID_PUBLISH',
            'DAYS_BIRTH',
            'DAYS_EMPLOYED'
        ]

        aggs = ['min', 'mean', 'max', 'sum', 'var']
        aggregation_pairs = [(col, agg) for col in cols_to_agg for agg in aggs]

        APPLICATION_AGGREGATION_RECIPIES = [
            (['NAME_EDUCATION_TYPE', 'CODE_GENDER'], aggregation_pairs),
            (['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], aggregation_pairs),
            (['NAME_FAMILY_STATUS', 'CODE_GENDER'], aggregation_pairs),
            (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                                    ('AMT_INCOME_TOTAL', 'mean'),
                                                    ('DAYS_REGISTRATION', 'mean'),
                                                    ('EXT_SOURCE_1', 'mean')]),
            (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                         ('CNT_CHILDREN', 'mean'),
                                                         ('DAYS_ID_PUBLISH', 'mean')]),
            (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'],
             [('EXT_SOURCE_1', 'mean'),
              ('EXT_SOURCE_2', 'mean')]),
            (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                          ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                          ('APARTMENTS_AVG', 'mean'),
                                                          ('BASEMENTAREA_AVG', 'mean'),
                                                          ('EXT_SOURCE_1', 'mean'),
                                                          ('EXT_SOURCE_2', 'mean'),
                                                          ('EXT_SOURCE_3', 'mean'),
                                                          ('NONLIVINGAREA_AVG', 'mean'),
                                                          ('OWN_CAR_AGE', 'mean'),
                                                          ('YEARS_BUILD_AVG', 'mean')]),
            (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                                    ('EXT_SOURCE_1', 'mean')]),
            (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                   ('CNT_CHILDREN', 'mean'),
                                   ('CNT_FAM_MEMBERS', 'mean'),
                                   ('DAYS_BIRTH', 'mean'),
                                   ('DAYS_EMPLOYED', 'mean'),
                                   ('DAYS_ID_PUBLISH', 'mean'),
                                   ('DAYS_REGISTRATION', 'mean'),
                                   ('EXT_SOURCE_1', 'mean'),
                                   ('EXT_SOURCE_2', 'mean'),
                                   ('EXT_SOURCE_3', 'mean')]),
        ]

        feats = GroupbyAggregateDiffs(APPLICATION_AGGREGATION_RECIPIES,
                                      use_diffs_only=True).aggregate(trn_tst)
        trn_tst = pd.concat([trn_tst[['SK_ID_CURR', 'is_tst']], feats], axis=1)

        del feats
        gc.collect()

        # setup for output
        trn_tst.columns = [
            "APPAGG_" + _c if _c not in ['SK_ID_CURR', 'is_tst'] else _c for _c in trn_tst.columns
        ]

        trn = trn_tst.query(
            'is_tst == False'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        tst = trn_tst.query(
            'is_tst == True'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        del trn_tst
        gc.collect()

        return trn, tst


if __name__ == "__main__":
    app_agg_trn, app_agg_tst = ApplicationAggregationAndDiff(const.FEATDIR).create_feature(None)
