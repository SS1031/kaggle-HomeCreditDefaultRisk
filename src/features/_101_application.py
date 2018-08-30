import gc
import pandas as pd
import numpy as np

import const
from features import FeatherFeatureTrnTstDF
from features._000_cleaning import application_cleaning
from features._999_utils import target_encode
from sklearn.model_selection import StratifiedKFold


class Application(FeatherFeatureTrnTstDF):

    @staticmethod
    def input():
        return const.in_app_trn, const.in_app_tst

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(trn: pd.DataFrame, tst: pd.DataFrame,
                                       random_state=None):
        y = trn.TARGET.copy()
        trn.drop(columns=['TARGET'], inplace=True)
        trn['is_tst'] = False
        tst['is_tst'] = True

        trn_tst = pd.concat([trn, tst], axis=0).reset_index(drop=True)

        trn_tst = application_cleaning(trn_tst)

        # 異常値replace
        trn_tst['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        # 年金 / 収入
        trn_tst['ANNUITY_TO_INCOME_RATIO'] = (
            trn_tst['AMT_ANNUITY'] / (1 + trn_tst['AMT_INCOME_TOTAL'])
        )

        # 車所持年齢 / 現年齢
        trn_tst['CAR_TO_BIRTH_RATIO'] = trn_tst['OWN_CAR_AGE'] / trn_tst['DAYS_BIRTH']

        # 車所持年齢 / 雇用年齢
        trn_tst['CAR_TO_EMPLOY_RATIO'] = trn_tst['OWN_CAR_AGE'] / trn_tst['DAYS_EMPLOYED']

        # 与信額 / 年金
        trn_tst['CREDIT_TO_ANNUITY_RATIO'] = trn_tst['AMT_CREDIT'] / trn_tst['AMT_ANNUITY']

        # 与信額 / グッズ額
        trn_tst['CREDIT_TO_GOODS_RATIO'] = trn_tst['AMT_CREDIT'] / trn_tst['AMT_GOODS_PRICE']

        # 収入額 / 子供人数
        trn_tst['CREDIT_PER_CHILD'] = trn_tst['AMT_CREDIT'] / (1 + trn_tst['CNT_CHILDREN'])

        # 子供以外の家族人数 = (家族人数 - 子供人数)
        trn_tst['CNT_NON_CHILD'] = trn_tst['CNT_FAM_MEMBERS'] - trn_tst['CNT_CHILDREN']

        # 収入額 / 子供以外の家族人数
        trn_tst['CREDIT_PER_NON_CHILD'] = trn_tst['AMT_CREDIT'] / trn_tst['CNT_NON_CHILD']

        # 子供人数 / 子供以外の家族人数
        trn_tst['CHILD_TO_NON_CHILD_RATIO'] = trn_tst['CNT_CHILDREN'] / trn_tst['CNT_NON_CHILD']

        # 雇用年齢 / 現年齢
        trn_tst['EMPLOY_TO_BIRTH_RATIO'] = trn_tst['DAYS_EMPLOYED'] / trn_tst['DAYS_BIRTH']

        # EXTERNAL_SOURCEを重み付けで加算
        trn_tst['EXTERNAL_SOURCES_WEIGHTED'] = (
            trn_tst.EXT_SOURCE_1 * 2 + trn_tst.EXT_SOURCE_2 * 3 + trn_tst.EXT_SOURCE_3 * 4
        )

        # 収入 / 与信額
        trn_tst['INCOME_CREDIT_PCT'] = trn_tst['AMT_INCOME_TOTAL'] / trn_tst['AMT_CREDIT']

        # 収入 / 子供の数30
        trn_tst['INCOME_PER_CHLD'] = trn_tst['AMT_INCOME_TOTAL'] / (1 + trn_tst['CNT_CHILDREN'])

        # 収入 / 子供以外の家族人数
        trn_tst['INCOME_PER_NON_CHILD'] = trn_tst['AMT_INCOME_TOTAL'] / trn_tst['CNT_NON_CHILD']

        # 収入 / 家族の人数
        trn_tst['INCOME_PER_PERSON'] = trn_tst['AMT_INCOME_TOTAL'] / trn_tst['CNT_FAM_MEMBERS']

        # 最後に機種変した日 / 現年齢
        trn_tst['PHONE_TO_BIRTH_RATIO'] = trn_tst['DAYS_LAST_PHONE_CHANGE'] / trn_tst['DAYS_BIRTH']

        # 最後に機種変した日 / 雇用年齢
        trn_tst['PHONE_TO_EMPLOY_RATIO'] = trn_tst['DAYS_LAST_PHONE_CHANGE'] / trn_tst['DAYS_EMPLOYED']

        inc_by_org = trn_tst[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
            'ORGANIZATION_TYPE'
        ).median()['AMT_INCOME_TOTAL']

        trn_tst['AVG_INCOME_BY_ORG'] = trn_tst['ORGANIZATION_TYPE'].map(inc_by_org)

        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            trn_tst['EXTERNAL_SOURCES_{}'.format(function_name)] = eval(
                'np.{}'.format(function_name)
            )(trn_tst[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        # setup for output
        trn_tst.columns = [
            "APP_" + _c if _c not in ['SK_ID_CURR', 'is_tst'] else _c for _c in trn_tst.columns
        ]

        trn = trn_tst.query(
            'is_tst == False'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        tst = trn_tst.query(
            'is_tst == True'
        ).copy().drop(columns=['is_tst']).reset_index(drop=True)

        del trn_tst
        gc.collect()

        # ターゲットエンコーディング
        target_encode_columns = ['APP_NAME_CONTRACT_TYPE', 'APP_CODE_GENDER',
                                 'APP_FLAG_OWN_CAR', 'APP_FLAG_OWN_REALTY',
                                 'APP_NAME_TYPE_SUITE', 'APP_NAME_INCOME_TYPE',
                                 'APP_NAME_EDUCATION_TYPE', 'APP_NAME_FAMILY_STATUS',
                                 'APP_NAME_HOUSING_TYPE', 'APP_OCCUPATION_TYPE',
                                 'APP_WEEKDAY_APPR_PROCESS_START', 'APP_ORGANIZATION_TYPE',
                                 'APP_FONDKAPREMONT_MODE', 'APP_HOUSETYPE_MODE',
                                 'APP_WALLSMATERIAL_MODE', 'APP_EMERGENCYSTATE_MODE']

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for col in target_encode_columns:
            print("TARGET_ENCODING,", col)
            trn[col + '_TE'] = 0.
            tst[col + '_TE'] = 0.
            SMOOTHING = tst[~tst[col].isin(trn[col])].shape[0] / tst.shape[0]
            for f, (vis_index, blind_index) in enumerate(kf.split(trn, y)):
                _, trn.loc[blind_index, col + '_TE'] = target_encode(
                    trn.loc[vis_index, col],
                    trn.loc[blind_index, col],
                    target=y.loc[vis_index],
                    min_samples_leaf=100,
                    smoothing=SMOOTHING,
                    noise_level=0.0
                )

                _, x = target_encode(
                    trn.loc[vis_index, col],
                    tst[col],
                    target=y.loc[vis_index],
                    min_samples_leaf=100,
                    smoothing=SMOOTHING,
                    noise_level=0.0)
                tst[col + '_TE'] += (0.2 * x)

            trn.drop(col, inplace=True, axis=1)
            tst.drop(col, inplace=True, axis=1)

        return trn, tst


if __name__ == '__main__':
    app101 = Application(const.FEATDIR).create_feature(None)
    app101df = pd.read_feather(app101[0])
