import numpy as np
import pandas as pd
import os
import sys


def load_dataset(dataset, *args, **kwargs):

    if dataset == 'kin40k':
        tx, ty, vx, vy, mx, my = load_kin40k(*args, **kwargs)
    elif dataset == 'abalone':
        tx, ty, vx, vy, mx, my = load_abalone(*args, **kwargs)
    elif dataset == 'boston':
        tx, ty, vx, vy, mx, my = load_boston(*args, **kwargs)
    elif dataset == 'sarcos':
        tx, ty, vx, vy, mx, my = load_sarcos(*args, **kwargs)
    elif dataset == 'mcycle':
        tx, ty, vx, vy, mx, my = load_mcycle(*args, **kwargs)
    else:
        return None

    assert len(tx) == len(ty)
    assert len(vx) == len(vy)
    assert len(mx) == len(my)

    tx, vx, mx = normalize(tx, vx, mx)
    ty, vy, my = normalize(ty, vy, my)

    print('%s loaded: %i train, %i val, %i test' % (
        dataset, len(tx), len(vx), len(mx)))
    print('Dimension of inputs: %i' % np.shape(tx)[1])

    return tx, ty, vx, vy, mx, my


def load_kin40k(val_prc=0.):

    kin40k_dir = '../data/kin40k/'

    def read_kin40k(fn):
        return pd.read_csv(
            kin40k_dir + fn, sep=' ',
            header=None, skipinitialspace=True).values

    tx = read_kin40k('kin40k_train_data.asc')
    ty = np.squeeze(read_kin40k('kin40k_train_labels.asc'))
    mx = read_kin40k('kin40k_test_data.asc')
    my = np.squeeze(read_kin40k('kin40k_test_labels.asc'))

    tx, ty, vx, vy = split_data(tx, ty, val_prc)

    return tx, ty, vx, vy, mx, my


def load_mcycle(val_prc=0.):

    df = pd.read_csv('../data/mcycle.csv')
    tx = df['times'].values[:, np.newaxis]
    ty = df['accel'].values

    tx, ty, mx, my = split_data(tx, ty, .2)
    tx, ty, vx, vy = split_data(tx, ty, val_prc)

    return tx, ty, vx, vy, mx, my


def load_sarcos():

    return None


def load_abalone(val_prc=0.):

    abalone_dir = '../data/abalone/'

    df = pd.read_csv(
        os.path.join(abalone_dir, 'abalone.data.txt'),
        header=None,
        names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
               'Shucked weight', 'Vicera weight', 'Shell weight', 'Rings'])

    df['Male'] = (df['Sex'] == 'M').astype(int)
    df['Female'] = (df['Sex'] == 'F').astype(int)
    df['Infant'] = (df['Sex'] == 'I').astype(int)

    df.drop('Sex', 1, inplace=True)

    x = df.drop('Rings', 1)
    y = df['Rings']

    mx = x[3133:].values
    tx = x[:3133].values

    my = y[3133:].values.astype(float)
    ty = y[:3133].values.astype(float)

    tx, ty, vx, vy = split_data(tx, ty, val_prc)

    return tx, ty, vx, vy, mx, my


def load_boston():

    return None


def normalize(*args):
    """normalize args based on mean and variance of args[0]"""

    vr = np.var(args[0], axis=0)
    mn = np.mean(args[0], axis=0)

    return ((y - mn) / vr for y in args)


def split_data(x, y, split_prc):

    if split_prc > 0.:

        v_indices = np.random.permutation(
            np.arange(len(x))) < int(split_prc * len(x))
        vx = x[v_indices, :]
        vy = y[v_indices]
        tx = x[~v_indices, :]
        ty = y[~v_indices]

    else:

        vx = None
        vy = None
        tx = x
        ty = y

    return tx, ty, vx, vy


if __name__ == '__main__':

    if(len(sys.argv) == 3):
        load_dataset(sys.argv[1], val_prc=sys.argv[2])
    elif(len(sys.argv) == 2):
        load_dataset(sys.argv[1], val_prc=.2)
    else:
        load_dataset('mcycle', val_prc=.2)
