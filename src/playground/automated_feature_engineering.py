# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

# matplotlit and seaborn for visualizations
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 22
import seaborn as sns

# Suppress warnings from pandas
import warnings

warnings.filterwarnings('ignore')

import const

app_train = pd.read_feather(const.in_app_trn).sort_values('SK_ID_CURR').reset_index(drop=True).loc[:1000, :]
app_test = pd.read_feather(const.in_app_tst).sort_values('SK_ID_CURR').reset_index(drop=True).loc[:1000, :]
bureau = pd.read_feather(const.in_bur).sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop=True).loc[:1000, :]
bureau_balance = pd.read_feather(const.in_bur).sort_values('SK_ID_BUREAU').reset_index(drop=True).loc[:1000, :]
cash = pd.read_feather(const.in_pcb).sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
credit = pd.read_feather(const.in_ccb).sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
previous = pd.read_feather(const.in_prv).sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
installments = pd.read_feather(const.in_inp).sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']
).reset_index(drop=True).loc[:1000, :]

app_train['set'] = 'train'
app_train['set'] = 'test'
app_test['TARGET'] = np.nan

# Append the dataframes
app = app_train.append(app_test, ignore_index=True)

# Entity set with id applications
es = ft.EntitySet(id='clients')

# Entities with a unique index
es = es.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR')
es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV')
# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id='bureau_balance', dataframe=bureau_balance,
                              make_index=True, index='bureaubalance_index')
es = es.entity_from_dataframe(entity_id='cash', dataframe=cash,
                              make_index=True, index='cash_index')
es = es.entity_from_dataframe(entity_id='installments', dataframe=installments,
                              make_index=True, index='installments_index')
es = es.entity_from_dataframe(entity_id='credit', dataframe=credit,
                              make_index=True, index='credit_index')

print('Parent: app, Parent Variable: SK_ID_CURR\n\n', app.iloc[:, 111:115].head())
print('\nChild: bureau, Child Variable: SK_ID_CURR\n\n', bureau.iloc[10:30, :4].head())

# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])
# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])
# Print out the EntitySet
print(es)

primitives = ft.list_primitives()
pd.options.display.max_colwidth = 1000
print(primitives[primitives['type'] == 'aggregation'].head(10))

default_agg_primitives = ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives = ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

# DFS with specified primitives
feature_names = ft.dfs(entityset=es, target_entity='app',
                       trans_primitives=default_trans_primitives,
                       agg_primitives=default_agg_primitives,
                       max_depth=2, features_only=True)

print('%d Total Features' % len(feature_names))
