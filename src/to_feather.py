import os
import pandas as pd
import config


def input_to_feather():
    files = [f for f in os.listdir(config.INDIR) if '.csv' in f]
    for f in files:
        print("to feather '{}'...".format(f))
        df = pd.read_csv(os.path.join(config.INDIR, f))
        df.to_feather(os.path.join(config.INDIR, f.split('.')[0] + '.feather'))


input_to_feather()
