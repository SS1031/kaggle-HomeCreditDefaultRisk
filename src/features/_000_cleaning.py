import numpy as np


def application_cleaning(app):
    app['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
    app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    app['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    app['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    app['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

    # 0しかないカラムを削除
    app.drop(columns=[
        'FLAG_DOCUMENT_2',
        'FLAG_DOCUMENT_10',
        'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_13',
        'FLAG_DOCUMENT_14',
        'FLAG_DOCUMENT_15',
        'FLAG_DOCUMENT_16',
        'FLAG_DOCUMENT_17',
        'FLAG_DOCUMENT_19',
        'FLAG_DOCUMENT_20',
        'FLAG_DOCUMENT_21'
    ], inplace=True)

    return app


def bureau_cleaning(bur, **kwargs):
    bur['DAYS_CREDIT_ENDDATE'][bur['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    bur['DAYS_CREDIT_UPDATE'][bur['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    bur['DAYS_ENDDATE_FACT'][bur['DAYS_ENDDATE_FACT'] < -40000] = np.nan

    if 'fill_value' in kwargs:
        bur['AMT_CREDIT_SUM'].fillna(kwargs['fill_value'], inplace=True)
        bur['AMT_CREDIT_SUM_DEBT'].fillna(kwargs['fill_value'], inplace=True)
        bur['AMT_CREDIT_SUM_OVERDUE'].fillna(kwargs['fill_value'], inplace=True)
        bur['CNT_CREDIT_PROLONG'].fillna(kwargs['fill_value'], inplace=True)

    return bur


def credit_card_balance_cleaning(ccb):
    ccb['AMT_DRAWINGS_ATM_CURRENT'][ccb['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    ccb['AMT_DRAWINGS_CURRENT'][ccb['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

    return ccb


def previous_application_cleaning(prv):
    prv['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prv['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prv['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prv['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prv['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    return prv
