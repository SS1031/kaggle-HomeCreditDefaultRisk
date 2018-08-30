import pandas as pd
import const


bur = pd.read_feather(const.in_bur)
bbl = pd.read_feather(const.in_bbl)

piv_bbl = bbl.pivot(index='SK_ID_BUREAU',
                    columns='MONTHS_BALANCE',
                    values='STATUS')


