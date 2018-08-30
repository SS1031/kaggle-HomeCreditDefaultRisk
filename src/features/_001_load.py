import gc
import json
import pprint
import itertools
import pandas as pd
import pandas.testing

from multiprocessing.pool import Pool

import const
# import features.basic

from features._101_application import Application
from features._102_application_aggregation import ApplicationAggregationAndDiff
from features._103_application_ext_source_knn import ApplicationEXTSourceKNN
from features._104_application_numeric_knn import ApplicationNumericKNN

from features._201_bureau import Bureau
from features._201_bureau_v2 import BureauV2
from features._202_bureau_aggregation import BureauAggregation
from features._202_bureau_aggregation_v2 import BureauAggregationV2

from features._301_bureau_balance_latest_status import BureauBlanceLatestStatusNorm
from features._302_bureau_balance_aggregation import BureauBalanceAggregation

from features._401_credit_card_balance import CreditCardBalance
from features._402_credit_card_balance_aggregation import CreditCardBalanceAggregation
from features._403_credit_card_balance_updown import CreditCardBalanceUpDown
from features._404_credit_card_balance_timeseries_lda import CreditCardBalanceTimeseriesLDA
from features._405_credit_card_balance_weighted_average import CreditCardBalanceWeightedAvg

from features._501_pos_cash_balance import POSCASHBalance
from features._502_pos_cash_balance_aggregation import POSCASHBalanceAggregation
from features._503_pos_cash_balance_timeseries_lda import POSCASHBalanceTimeseriesLDA
from features._504_pos_cash_balance_weighted_average import POSCASHBalanceWeightedAvg

from features._601_previous_application import PreviousApplication
from features._602_previous_application_aggregation import PreviousApplicationAggregation
from features._603_pervious_application_timeseries import PreviousApplicationTimeseriesLDA

from features._701_installment_payments import InstallmentPayments
from features._702_installment_payments_aggregation import InstallmentPaymentsAggregation
from features._703_installment_payments_firstk_aggregation import InstallmentPaymentsFirstKAggregation
from features._704_installment_payments_dae import InstallmentPaymentsTimeseriesDAE
from features._705_installment_payments_weighted_average import InstallmentPaymentsWeightedAverage

from features._002_feature_selection import remove_high_correlation_column
from features._002_feature_selection import remove_high_missing_column
from features._002_feature_selection import remove_low_importance_feature
from features._002_feature_selection import null_importances_feature_selection

# 並列計算できる乱数使わない特徴量
para_feature_map = {
    # 'app': features.basic.App,
    'application_aggregation': ApplicationAggregationAndDiff,
    'bureau': Bureau,
    'bureau_v2': BureauV2,
    'bureau_aggregation': BureauAggregation,
    'bureau_aggregation_v2': BureauAggregationV2,
    'bureau_balance_latest_status': BureauBlanceLatestStatusNorm,
    'bureau_balance_aggregation': BureauBalanceAggregation,
    'credit_card_balance': CreditCardBalance,
    'credit_card_balance_aggregation': CreditCardBalanceAggregation,
    'credit_card_balance_updown': CreditCardBalanceUpDown,
    'pos_cash_balance_aggregation': POSCASHBalanceAggregation,
    'previous_application_aggregation': PreviousApplicationAggregation,
    'installment_payments_aggregation': InstallmentPaymentsAggregation,
    'installment_payments_firstk_aggregation': InstallmentPaymentsFirstKAggregation,
}

# 並列計算できる乱数使う特徴量
para_feature_map_with_rand = {
    'application': Application,
}

unpara_feature_map = {
    'credit_card_balance_weighted_average': CreditCardBalanceWeightedAvg,
    'pos_cash_balance_weighted_average': POSCASHBalanceWeightedAvg,
    'installment_payments_weighted_average': InstallmentPaymentsWeightedAverage,
    'previous_application': PreviousApplication,
    'pos_cash_balance': POSCASHBalance,
    'installment_payments': InstallmentPayments,
    'installment_payments_timeseries_dae': InstallmentPaymentsTimeseriesDAE
}

unpara_feature_map_with_rand = {
    'application_ext_source_knn': ApplicationEXTSourceKNN,
    'application_numeric_knn': ApplicationNumericKNN,
    'previous_application_timeseries_lda': PreviousApplicationTimeseriesLDA,
    'credit_card_balance_timeseries_lda': CreditCardBalanceTimeseriesLDA,
    'pos_cash_balance_timeseries_lda': POSCASHBalanceTimeseriesLDA,
}


def get_feature(feature_name):
    cache_dir = const.FEATDIR
    if isinstance(feature_name, list):
        feature_name = feature_name[0]

    if feature_name in para_feature_map:
        return para_feature_map[feature_name](cache_dir)

    if feature_name in para_feature_map_with_rand:
        return para_feature_map_with_rand[feature_name](cache_dir)

    if feature_name in unpara_feature_map:
        return unpara_feature_map[feature_name](cache_dir)

    if feature_name in unpara_feature_map_with_rand:
        return unpara_feature_map_with_rand[feature_name](cache_dir)


def load_feature(arg):
    feature_name, random_seed, args = arg

    feature = get_feature(feature_name=feature_name)

    return feature.create_feature(random_seed, *args)


def load_features(config_features, random_state):
    print("Loading features ...")

    para_feature_args = []
    unpara_feature_args = []

    for feat in config_features['use_features']:

        if isinstance(feat, list):
            fname = feat[0]
            args = feat[1]
        else:
            fname = feat
            args = {}

        if fname in para_feature_map:
            para_feature_args.append((fname, None, args))

        if fname in para_feature_map_with_rand:
            para_feature_args.append((fname, random_state, args))

        if fname in unpara_feature_map:
            unpara_feature_args.append((fname, None, args))

        if fname in unpara_feature_map_with_rand:
            unpara_feature_args.append((fname, random_state, args))

    with Pool(4) as p:
        responses = p.map(load_feature, para_feature_args)

    for feat in unpara_feature_args:
        responses.append(load_feature(feat))

    trn_feats = []
    tst_feats = []
    id_feats = []

    for res in responses:

        if isinstance(res, tuple):
            trn_feats.append([r for r in res if 'trn' in r][0])
            tst_feats.append([r for r in res if 'tst' in r][0])
        else:
            id_feats.append(res)

    return trn_feats, tst_feats, id_feats


def load_categorical_features(config_features):
    categorical_features = []
    for feature in config_features['use_features']:
        if isinstance(feature, list):
            fname = feature[0]
        else:
            fname = feature
        categorical_features += get_feature(fname).categorical_features()
    # return list(itertools.chain(*[
    #     get_feature(feature).categorical_features() for feature in
    # ]))

    return categorical_features


def dataset(paths, id_feat_paths):
    assert len(paths) > 0

    feature_datasets = []
    for path in paths:
        feature_datasets.append(pd.read_feather(path).set_index('SK_ID_CURR'))

    # check if all of feature dataset share the same index
    index = feature_datasets[0].index
    for feature_dataset in feature_datasets[1:]:
        pandas.testing.assert_index_equal(index, feature_dataset.index)

    feature_datasets = pd.concat(feature_datasets, axis=1)

    # add TARGET columns into train dataset
    if 'trn' in paths[0]:
        target = pd.read_feather('../data/input/application_train.feather')[
            ['SK_ID_CURR', 'TARGET']
        ].set_index('SK_ID_CURR')
        pandas.testing.assert_index_equal(index, target.index)
        feature_datasets = pd.concat([feature_datasets, target], axis=1)

    feature_datasets.reset_index(inplace=True)
    for id_feat_path in id_feat_paths:
        feature_datasets = feature_datasets.merge(pd.read_feather(id_feat_path),
                                                  how='left', on='SK_ID_CURR')

    return feature_datasets


def selection(trn, tst, config_selection):
    print("Data shape before selection")
    print("     Train shape : {}".format(trn.shape))
    print("     Test shape  : {}".format(tst.shape))

    target = trn.TARGET.copy()

    trn_sk_id_curr = trn.SK_ID_CURR.copy()
    tst_sk_id_curr = tst.SK_ID_CURR.copy()

    trn.drop(columns=['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    tst.drop(columns=['SK_ID_CURR'], axis=1, inplace=True)

    for conf in config_selection:
        if conf['method'] == "missing":
            trn, tst = remove_high_missing_column(trn, tst, conf['threshold'])
        if conf['method'] == "correlation":
            trn, tst = remove_high_correlation_column(trn, tst, conf['threshold'])
        if conf['method'] == "lgb_importance":
            trn, tst = remove_low_importance_feature(trn, target, tst,
                                                     conf['iteration'], conf['threshold'])
        if conf['method'] == "null_importances":
            trn, tst = null_importances_feature_selection(trn, target, tst)

    trn['TARGET'] = target
    trn['SK_ID_CURR'] = trn_sk_id_curr
    tst['SK_ID_CURR'] = tst_sk_id_curr

    print("Final data shape after selection")
    print("     Train shape : {}".format(trn.shape))
    print("     Test shape  : {}".format(tst.shape))

    return trn, tst


def datasets(config_features, random_state, debug=False):
    trn_feat_paths, tst_feat_paths, id_feat_paths = load_features(config_features, random_state)
    categorical_features = load_categorical_features(config_features)

    trn = dataset(trn_feat_paths, id_feat_paths)
    tst = dataset(tst_feat_paths, id_feat_paths)

    for cat in categorical_features:
        trn[cat] = trn[cat].astype('category')
        tst[cat] = tst[cat].astype('category')

    # feature selection
    trn, tst = selection(trn, tst, config_features['selection'])

    # == Trueにしないと効かない、なぜ??こわい
    if debug == True:
        print("DEBUG mode")
        trn = trn.iloc[:10000]

    return trn, tst, categorical_features


if __name__ == "__main__":
    import argparse
    from importlib import import_module
    from collections import OrderedDict

    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm16.json')

    options = parser.parse_args()
    conf = json.load(open(options.config), object_pairs_hook=OrderedDict)

    pp = pprint.PrettyPrinter(indent=2)
    print("Settings : ")
    pp.pprint(conf)
    trn, tst, categorical_features = datasets(conf['feature'], debug=False)
