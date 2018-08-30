import time
import hashlib
import os
import gc

from abc import ABC, abstractmethod

from typing import List, Tuple
import pandas as pd
import json


class Feature(ABC):

    def __init__(self, features_dir):
        self.features_dir = features_dir

    @property
    def name(self) -> str:
        return self.__class__.__name__


class FeatureTrnTst(Feature):
    """trainとtestに別れる特徴量
    """

    @staticmethod
    @abstractmethod
    def input() -> Tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def create_feature(self, random_state) -> Tuple[List[str], str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def categorical_features():
        raise NotImplementedError


class FeatureID(Feature):
    """SK_ID_CURRに対して作成する特徴量
    """

    @staticmethod
    @abstractmethod
    def input() -> str:
        raise NotImplementedError

    @abstractmethod
    def create_feature(self, random_state) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def categorical_features():
        raise NotImplementedError


class FeatherFeatureTrnTst(FeatureTrnTst):

    def create_feature(self, random_state, *args) -> (List[str], str):
        trn_path, tst_path = self.input()

        trn_feature_path = self.get_feature_file('trn', trn_path,
                                                 tst_path, random_state=random_state)

        tst_feature_path = self.get_feature_file('tst', trn_path,
                                                 tst_path, random_state=random_state)

        if os.path.exists(trn_feature_path) and os.path.exists(tst_feature_path):
            print("There are cache files for feature [{}] (trn_path=[{}], tst_path=[{}])"
                  .format(self.name, trn_path, tst_path))
            return trn_feature_path, tst_feature_path

        print("Start computing feature [{}] (trn_path=[{}], tst_path=[{}])".format(
            self.name, trn_feature_path, tst_feature_path
        ))
        start_time = time.time()
        self.create_features_impl(trn_path=trn_path,
                                  tst_path=tst_path,
                                  trn_feature_path=trn_feature_path,
                                  tst_feature_path=tst_feature_path,
                                  random_state=random_state)

        print("Finished computing feature [{}] (trn_path=[{}], tst_path=[{}]): {:.3} [s]"
              .format(self.name, trn_path, tst_path, time.time() - start_time))

        return trn_feature_path, tst_feature_path

    def get_feature_file(self, dataset_type: str, trn_path: str,
                         tst_path: str, random_state) -> str:
        feature_cache_suffix = self.get_feature_suffix(trn_path, tst_path, random_state)
        filename = self.name + '_' + dataset_type + '_' + feature_cache_suffix + '.feather'

        return os.path.join(self.features_dir, filename)

    @staticmethod
    def get_feature_suffix(trn_path: str, tst_path, random_state) -> str:

        if random_state is None:
            return (hashlib.md5(str([trn_path, tst_path]).encode('utf-8')).hexdigest()[:10])
        else:
            return (hashlib.md5(str([trn_path, tst_path]).encode('utf-8')).hexdigest()[:10] +
                    "_{}".format(random_state))

    @abstractmethod
    def create_features_impl(self, trn_path: str, tst_path: str,
                             trn_feature_path: str, tst_feature_path: str,
                             random_state: int):
        raise NotImplementedError


class FeatherFeatureID(FeatureID):

    def create_feature(self, random_state, *args) -> (List[str], str):

        path = self.input()
        if len(args) > 0:
            feature_path = self.get_feature_file(path, random_state, *args)
        else:
            feature_path = self.get_feature_file(path, random_state)

        if os.path.exists(feature_path):
            print("There are cache files for feature [{}] (path=[{}])".format(self.name,
                                                                              feature_path))
            return feature_path

        print("Start computing feature [{}] (path=[{}])".format(
            self.name, feature_path
        ))
        start_time = time.time()
        self.create_features_impl(path,
                                  feature_path,
                                  random_state,
                                  *args)

        print("Finished computing feature [{}] (path=[{}]): {:.3} [s]"
              .format(self.name, path, time.time() - start_time))

        return feature_path

    def get_feature_file(self, path: str, random_state, *args) -> str:
        feature_cache_suffix = self.get_feature_suffix(path, random_state, *args)
        filename = self.name + '_' + feature_cache_suffix + '.feather'

        return os.path.join(self.features_dir, filename)

    @staticmethod
    def get_feature_suffix(path, random_state, *args) -> str:
        str_args = ""
        if len(args) > 0:
            for a in args:
                str_args += str(a).replace(" ", "")

        if random_state is None:
            return (
                hashlib.md5((path + str_args).encode('utf-8')).hexdigest()[:10]
            )
        else:
            return (
                hashlib.md5(
                    (path + str_args).encode('utf-8')
                ).hexdigest()[:10] + "_{}".format(random_state)
            )

    @abstractmethod
    def create_features_impl(self, path: str, feature_path: str, random_state, *args):
        raise NotImplementedError


class FeatherFeatureTrnTstDF(FeatherFeatureTrnTst):

    def create_features_impl(self, trn_path: str, tst_path: str,
                             trn_feature_path: str, tst_feature_path: str,
                             random_state, *args):
        trn = pd.read_feather(trn_path)
        tst = pd.read_feather(tst_path)

        trn_feature, tst_feature = self.create_features_from_dataframe(trn, tst, random_state)

        trn_feature.to_feather(trn_feature_path)
        tst_feature.to_feather(tst_feature_path)

    @staticmethod
    def create_features_from_dataframe(trn: pd.DataFrame, tst: pd.DataFrame,
                                       random_state):
        raise NotImplementedError


class FeatherFeatureIDFromDF(FeatherFeatureID):

    def create_features_impl(self, path: str, feature_path: str, random_state, *args):
        df = pd.read_feather(path)

        feature = self.create_features_from_dataframe(df, random_state, *args)
        feature.to_feather(feature_path)

    @staticmethod
    def create_features_from_dataframe(df: pd.DataFrame, random_state, *args):

        raise NotImplementedError
