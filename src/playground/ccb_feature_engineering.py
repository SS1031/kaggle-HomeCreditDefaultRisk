import pandas as pd
import numpy as np
import const
import matplotlib.pyplot as plt
import seaborn as sns


def target_hist(df, col, title):
    df0 = df[df["TARGET"] == 0]
    df1 = df[df["TARGET"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.distplot(df0[col].dropna(), ax=axes[0], color='b', kde=True, norm_hist=True)
    axes[0].set_title('Target=0')
    sns.distplot(df1[col].dropna(), ax=axes[1], color='orange', kde=True, norm_hist=True)
    axes[1].set_title('Target=1')
    fig.suptitle(title)
    plt.show()


# AMT_BALANCE
def cnt_up(grp):
    return (grp.diff() > 0).sum()


def cnt_down(grp):
    return (grp.diff() < 0).sum()


def mean_up(grp):
    # 増大額の平均
    val = grp.diff()[grp.diff() > 0].mean()
    return val if val == val else 0


def mean_down(grp):
    # 減少額の平均
    val = grp.diff()[grp.diff() < 0].mean()
    return val if val == val else 0


# drawingのカウント
def cnt_drawings(grp):
    return (grp > 0).sum()


def cnt_minus(grp):
    return (grp < 0).sum()


ccb = pd.read_feather(const.in_ccb)
app_trn = pd.read_feather(const.in_app_trn)

# 前処理
ccb = ccb.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE']).reset_index(drop=True)
# 借入総額 / 借入上限
ccb['CREDIT_RATE'] = ccb['AMT_BALANCE'] / ccb['AMT_CREDIT_LIMIT_ACTUAL']
ccb['CREDIT_RATE'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
ccb['DRAWINGS_ATM_RATE'] = ccb['AMT_DRAWINGS_ATM_CURRENT'] / ccb['AMT_DRAWINGS_CURRENT']
ccb['DRAWINGS_ATM_RATE'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
# 実際の支払額(installment) / 最低支払い規定額(installment)
ccb['INST_PAYMENT_CURRENT_RATE'] = (
    ccb['AMT_PAYMENT_TOTAL_CURRENT'] / ccb['AMT_INST_MIN_REGULARITY']
)
ccb['INST_PAYMENT_CURRENT_RATE'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# 実際の支払額(installment) - 最低支払い規定額(installment),
# 最低支払い規定額を下回った回数のカウントを行う
ccb['INST_PAYMENT_CURRENT_DIFF'] = (
    ccb['AMT_PAYMENT_TOTAL_CURRENT'] - ccb['AMT_INST_MIN_REGULARITY']
)

# NAME_CONTRACT_STATUSのダミー化
dum_name_contract_status = pd.get_dummies(ccb.NAME_CONTRACT_STATUS)
dum_name_contract_status_cols = [
    'NAME_CONTRACT_STATUS_{}'.format(
        col.replace(' ', '')
    ) for col in dum_name_contract_status.columns
]
dum_name_contract_status.columns = dum_name_contract_status_cols
ccb = pd.concat([ccb, dum_name_contract_status], axis=1)

# aggregation方法のdictを作成
dum_prv_agg_dict = {}
for col in dum_name_contract_status_cols:
    dum_prv_agg_dict[col] = 'sum'

# aggregation方法のdictを作成
ccb_prv_agg_dict = {
    'SK_ID_CURR': [pd.Series.unique, 'size'],
    'AMT_BALANCE': ['max', cnt_up, cnt_down, mean_up, mean_down],
    'AMT_CREDIT_LIMIT_ACTUAL': ['mean', pd.Series.nunique],
    'CREDIT_RATE': ['max', 'mean', 'min', 'var'],
    'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'var', cnt_drawings],
    'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'var', cnt_drawings],
    'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'var', cnt_drawings],
    'INST_PAYMENT_CURRENT_RATE': ['max', 'min', 'mean', 'var'],
    'INST_PAYMENT_CURRENT_DIFF': [cnt_minus],
    'CNT_DRAWINGS_CURRENT': ['sum', 'max'],
    'CNT_DRAWINGS_ATM_CURRENT': ['sum', 'max'],
    'CNT_DRAWINGS_POS_CURRENT': ['sum', 'max'],
    'CNT_DRAWINGS_OTHER_CURRENT': ['sum', 'max'],
    'CNT_INSTALMENT_MATURE_CUM': 'max',
}
ccb_prv_agg_dict.update(dum_prv_agg_dict)

ccb_prv_agg = ccb.groupby('SK_ID_PREV').agg(ccb_prv_agg_dict).fillna(0)


# カラム名の変更
ccb_prv_agg.columns = ['_'.join(col).strip() for col in ccb_prv_agg.columns.values]

# SK_ID_CURRのrename
ccb_prv_agg.rename(columns={'SK_ID_CURR_unique': 'SK_ID_CURR'}, inplace=True)


# AMT_BALANCEが上昇した割合
ccb_prv_agg['AMT_BALANCE_UP_RATIO'] = (
    ccb_prv_agg.AMT_BALANCE_cnt_up / ccb_prv_agg.SK_ID_CURR_size
)

ccb_prv_agg.reset_index(inplace=True)
ccb_cur_agg = ccb_prv_agg.groupby('SK_ID_CURR')[
    [col for col in ccb_prv_agg.columns if col not in ['SK_ID_PREV', 'SK_ID_CURR']]
].max()
ccb_cur_agg['SK_ID_PREV_size'] = ccb_prv_agg.groupby('SK_ID_CURR').SK_ID_PREV.size()
