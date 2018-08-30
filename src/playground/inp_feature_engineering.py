import pandas as pd
import numpy as np
import const
import matplotlib.pyplot as plt
import seaborn as sns


def target_hist(df, col, title):
    df0 = df[df["TARGET"] == 0]
    df1 = df[df["TARGET"] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.distplot(df0[col].dropna(), ax=axes[0], color='b')
    axes[0].set_title('Target=0')
    sns.distplot(df1[col].dropna(), ax=axes[1], color='orange')
    axes[1].set_title('Target=1')
    fig.suptitle(title)
    plt.show()


inp = pd.read_feather(const.in_inp)
app_trn = pd.read_feather(const.in_app_trn)

# Days Before Due date
inp['DBD'] = -(inp['DAYS_ENTRY_PAYMENT'] - inp['DAYS_INSTALMENT'])
# 実際の支払い金額 / 支払い予定額 = RAP(Rate Amt Payment)
inp['RAP'] = inp['AMT_PAYMENT'] / inp['AMT_INSTALMENT']
inp['RAP'].replace([np.inf, -np.inf], np.nan, inplace=True)
# 支払い予定額を下回ったかどうか = FBI(Flag Below Instalment)
inp['FBI'] = inp['AMT_PAYMENT'] < inp['AMT_INSTALMENT']

prv_agg = inp.groupby('SK_ID_PREV').agg({
    'SK_ID_CURR': pd.Series.unique,
    'NUM_INSTALMENT_VERSION': pd.Series.nunique,
    'NUM_INSTALMENT_NUMBER': 'max',
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'min', 'var'],
    'DAYS_INSTALMENT': ['max', 'mean', 'min', 'var'],
    'DBD': ['max', 'mean', 'min', 'var'],
    'AMT_PAYMENT': ['max', 'mean', 'min', 'var'],
    'AMT_INSTALMENT': ['max', 'mean', 'min', 'var'],
    'RAP': ['max', 'mean', 'min', 'var'],
    'FBI': 'sum',
})
prv_agg.columns = ['_'.join(col).strip() for col in prv_agg.columns.values]

prv_agg.rename(columns={
    'AMT_INSTALMENT_max': 'AMT_INSTALMENT_MAX',
    'AMT_INSTALMENT_mean': 'AMT_INSTALMENT_AVG',
    'AMT_INSTALMENT_min': 'AMT_INSTALMENT_MIN',
    'AMT_INSTALMENT_var': 'AMT_INSTALMENT_VAR',
    'DAYS_INSTALMENT_max': 'DAYS_INSTALMENT_MAX',
    'DAYS_INSTALMENT_mean': 'DAYS_INSTALMENT_AVG',
    'DAYS_INSTALMENT_min': 'DAYS_INSTALMENT_MIN',
    'DAYS_INSTALMENT_var': 'DAYS_INSTALMENT_VAR',
    'FBI_sum': 'FBI_SUM',
    'RAP_max': 'RAP_MAX',
    'RAP_mean': 'RAP_MEAN',
    'RAP_min': 'RAP_MIN',
    'RAP_var': 'RAP_VAR',
    'NUM_INSTALMENT_VERSION_nunique': 'NUM_INSTALMENT_VERSION_CNT',
    'NUM_INSTALMENT_NUMBER_max': 'NUM_INSTLAMENT_NUMBER_MAX',
    'DAYS_ENTRY_PAYMENT_max': 'DAYS_ENTRY_PAYMENT_MAX',
    'DAYS_ENTRY_PAYMENT_mean': 'DAYS_ENTRY_PAYMENT_AVG',
    'DAYS_ENTRY_PAYMENT_min': 'DAYS_ENTRY_PAYMENT_MIN',
    'DAYS_ENTRY_PAYMENT_var': 'DAYS_ENTRY_PAYMENT_VAR',
    'SK_ID_CURR_unique': 'SK_ID_CURR',
    'AMT_PAYMENT_max': 'AMT_PAYMENT_MAX',
    'AMT_PAYMENT_mean': 'AMT_PAYMENT_AVG',
    'AMT_PAYMENT_min': 'AMT_PAYMENT_MIN',
    'AMT_PAYMENT_var': 'AMT_PAYMENT_VAR',
    'DBD_max': 'DBD_MAX',
    'DBD_mean': 'DBD_MEAN',
    'DBD_min': 'DBD_MIN',
    'DBD_var': 'DBD_VAR'
}, inplace=True)

prv_agg.reset_index(drop=True, inplace=True)

agg_cols = [
    'AMT_INSTALMENT_MAX',
    'AMT_INSTALMENT_AVG',
    'AMT_INSTALMENT_MIN',
    'AMT_INSTALMENT_VAR',
    'DAYS_INSTALMENT_MAX',
    'DAYS_INSTALMENT_AVG',
    'DAYS_INSTALMENT_MIN',
    'DAYS_INSTALMENT_VAR',
    'FBI_SUM',
    'RAP_MAX',
    'RAP_MEAN',
    'RAP_MIN',
    'RAP_VAR',
    'NUM_INSTALMENT_VERSION_CNT',
    'NUM_INSTLAMENT_NUMBER_MAX',
    'DAYS_ENTRY_PAYMENT_MAX',
    'DAYS_ENTRY_PAYMENT_AVG',
    'DAYS_ENTRY_PAYMENT_MIN',
    'DAYS_ENTRY_PAYMENT_VAR',
    'AMT_PAYMENT_MAX',
    'AMT_PAYMENT_AVG',
    'AMT_PAYMENT_MIN',
    'AMT_PAYMENT_VAR',
    'DBD_MAX',
    'DBD_MEAN',
    'DBD_MIN',
    'DBD_VAR'
]

inp_agg = prv_agg.groupby('SK_ID_CURR').size().rename('INP_CNT(SK_ID_CURR)')
for col in agg_cols:
    part_agg = prv_agg.groupby('SK_ID_CURR')[col].agg(['mean', 'max', 'min', 'var']).fillna(0)
    part_agg.columns = [
        'INP_{}({})'.format(name_stat, col) for name_stat in ['AVG', 'MAX', 'MIN', 'VAR']
    ]
    inp_agg = pd.concat([inp_agg, part_agg], axis=1)

inp_agg.reset_index(inplace=True)
inp_agg = inp_agg.merge(app_trn[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='inner')

target_hist(inp_agg, 'INP_AVG(DBD_MAX)', 'INP_AVG(DBD_MAX) by Target')
