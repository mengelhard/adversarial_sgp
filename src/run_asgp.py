import numpy as np
from load_data import load_dataset
from ASGP import learn_asgp
import pandas as pd
import datetime
import sys


def kin40k_experiments():

    tx, ty, vx, vy, mx, my = load_dataset('kin40k', val_prc=.2)

    results = []

    for num_ind_inputs in [32, 64, 128, 256, 512, 1024]:

        for gaussian_reference in [True, False]:

            for g_lr in [1e-1, 1e-2, 1e-3]:

                for mw_alpha in [1., 10., 100.]:

                    print('Starting New Run:')

                    try:

                        gloss_all, val_nmse_all, test_nmse_ = learn_asgp(
                            tx, ty, vx, vy, mx, my,
                            num_ind_inputs=num_ind_inputs,
                            gaussian_reference=gaussian_reference,
                            mw_alpha=mw_alpha,
                            g_lr=g_lr)

                        results.append({
                            'num_ind_inputs': num_ind_inputs,
                            'mw_alpha': mw_alpha,
                            'gaussian_reference': gaussian_reference,
                            'complete': True,
                            'gloss_start': gloss_all[0],
                            'gloss_final': gloss_all[-1],
                            'gloss_best': np.amin(gloss_all),
                            'val_nmse_start': val_nmse_all[0],
                            'val_nmse_final': val_nmse_all[-1],
                            'val_nmse_best': np.amin(val_nmse_all),
                            'test_nmse': test_nmse_})

                    except Exception as e:

                        print(e)

                        results.append({
                            'num_ind_inputs': num_ind_inputs,
                            'mw_alpha': mw_alpha,
                            'gaussian_reference': gaussian_reference,
                            'complete': False,
                            'error_message': e})

    fn = datetime.datetime.now().strftime('%y%m%d_%H:%M')
    pd.DataFrame(results).to_csv('../results/kin40k_' + fn + '.csv')


if __name__ == '__main__':

    if sys.argv[1] == 'kin40k':
        kin40k_experiments()
