import os
import json
import pandas as pd

import numpy as np
from datetime import datetime


def dump_json_log(options, train_results, output_directory):
    config = json.load(open(options.config))
    results = {
        'training': {
            'trials': train_results,
            'average_train_auc': np.mean([result['train_auc'] for result in train_results]),
            'average_valid_auc': np.mean([result['valid_auc'] for result in train_results]),
            'train_auc_std': np.std([result['train_auc'] for result in train_results]),
            'valid_auc_std': np.std([result['valid_auc'] for result in train_results]),
            'average_train_time': np.mean([result['train_time'] for result in train_results])
        },
        'config': config,
    }
    log_path = os.path.join(output_directory,
                            os.path.splitext(os.path.basename(options.config))[0] + '.result.json')
    json.dump(results, open(log_path, 'w'), indent=2)


def dump_json_opt_log(bst_params, trials, output_path):

    ret = {
        'bst_params': bst_params,
        'trials': trials
    }


def update_result_summary(run_file, options, train_results):

    df = pd.DataFrame()

    df['datetime'] = [datetime.now().strftime('%Y/%m/%d %H:%M:%S')]
    df['run_file'] = [run_file]  # os.path.basename(__file__)
    df['config_file'] = [options.config]
    df['average_train_auc'] = [np.mean([result['train_auc'] for result in train_results])]
    df['average_valid_auc'] = [np.mean([result['valid_auc'] for result in train_results])]
    df['train_auc_std'] = [np.std([result['train_auc'] for result in train_results])]
    df['valid_auc_std'] = [np.std([result['valid_auc'] for result in train_results])]

    result_file = '../data/output/result_summary.csv'
    if os.path.exists(result_file):
        ret_df = pd.read_csv(result_file)
        ret_df = pd.concat([ret_df, df], axis=0)
    else:
        ret_df = df

    ret_df.to_csv(result_file, index=False)
