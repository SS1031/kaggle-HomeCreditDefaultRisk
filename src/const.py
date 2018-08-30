import os


INDIR = '../data/input'
OUTDIR = '../data/output'
FEATDIR = '../data/features'

in_app_trn = os.path.join(INDIR, 'application_train.feather')
in_app_tst = os.path.join(INDIR, 'application_test.feather')
in_bur = os.path.join(INDIR, 'bureau.feather')
in_bbl = os.path.join(INDIR, 'bureau_balance.feather')
in_pcb = os.path.join(INDIR, 'POS_CASH_balance.feather')
in_ccb = os.path.join(INDIR, 'credit_card_balance.feather')
in_prv = os.path.join(INDIR, 'previous_application.feather')
in_inp = os.path.join(INDIR, 'installments_payments.feather')

TARGET_COL = 'TARGET'
