import seaborn as sns

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import const

print(os.listdir("../input"))
pd.options.mode.chained_assignment = None

train_df = pd.read_feather(const.in_app_trn)
test_df = pd.read_feather(const.in_app_tst)
sample_subm = pd.read_csv('../data/input/sample_submission.csv')

train_df_ltd = train_df[[
    'AMT_INCOME_TOTAL',
    'AMT_ANNUITY', 'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'TARGET',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]]

test_df_ltd = test_df[[
    'AMT_INCOME_TOTAL',
    'AMT_ANNUITY',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]]

y = train_df_ltd['TARGET']
del train_df_ltd['TARGET']

# let's add one feature that we know performs very well
train_df_ltd['ann_nr'] = train_df_ltd['AMT_CREDIT'] / train_df_ltd['AMT_ANNUITY']
test_df_ltd['ann_nr'] = test_df_ltd['AMT_CREDIT'] / test_df_ltd['AMT_ANNUITY']

pca = PCA()
standard_scaler = StandardScaler()
train_scaled = standard_scaler.fit_transform(train_df_ltd.fillna(-1))
pca.fit(train_scaled)

pca.explained_variance_