import numpy as np
import pandas as pd
import os

def load_dataset(dataset,*args,**kwargs):
    if dataset == 'kin40k':
        return load_kin40k(*args,**kwargs)
    elif dataset == 'abalone':
        return load_abalone(*args,**kwargs)
    elif dataset == 'boston':
        return load_boston(*args,**kwargs)
    elif dataset == 'sarcos':
        return load_sarcos(*args,**kwargs)
    else:
        return None

def load_kin40k(val_prc=0.):

    kin40k_dir = '~/sgp/data/kin40k/'

    # Read KIN40K Data
    train_x = pd.read_csv(kin40k_dir+'kin40k_train_data.asc',sep=' ',
        header=None,skipinitialspace=True).values
    train_y = pd.read_csv(kin40k_dir+'kin40k_train_labels.asc',sep=' ',
        header=None,skipinitialspace=True).values
    test_x = pd.read_csv(kin40k_dir+'kin40k_test_data.asc',sep=' ',
        header=None,skipinitialspace=True).values
    test_y = pd.read_csv(kin40k_dir+'kin40k_test_labels.asc',sep=' ',
        header=None,skipinitialspace=True).values
    train_y = np.squeeze(train_y)
    test_y = np.squeeze(test_y)

    # Normalize KIN40K Data

    x_var = np.var(train_x,axis=0)
    x_mean = np.mean(train_x,axis=0)
    y_var = np.var(train_y)
    y_mean = np.mean(train_y)

    train_x = (train_x-x_mean)/x_var
    test_x = (test_x-x_mean)/x_var
    train_y = (train_y-y_mean)/y_var
    test_y = (test_y-y_mean)/y_var

    # Create a validation set (20% of training data)

    val_indices = np.random.permutation(np.arange(len(train_x)))<int(val_prc*len(train_x))
    val_x = train_x[val_indices,:]
    val_y = train_y[val_indices]
    train_x = train_x[~val_indices,:]
    train_y = train_y[~val_indices]

    return train_x, train_y, val_x, val_y, test_x, test_y

def load_sarcos():

    return None

def load_abalone(val_prc=0.):

    abalone_dir = '~/sgp/data/abalone/'

    df = pd.read_csv(os.path.join(abalone_dir,'abalone.data.txt'),header=None,
        names=['Sex','Length','Diameter','Height','Whole weight','Shucked weight',
        'Vicera weight','Shell weight','Rings'])

    df['Male'] = (df['Sex'] == 'M').astype(int)
    df['Female'] = (df['Sex'] == 'F').astype(int)
    df['Infant'] = (df['Sex'] == 'I').astype(int)

    df.drop('Sex',1,inplace=True)

    x = df.drop('Rings',1)
    y = df['Rings']

    test_x = x[3133:].values
    train_x = x[:3133].values

    test_y = y[3133:].values.astype(float)
    train_y = y[:3133].values.astype(float)

    # Normalize Data

    x_var = np.var(train_x,axis=0)
    x_mean = np.mean(train_x,axis=0)
    y_var = np.var(train_y)
    y_mean = np.mean(train_y)

    train_x = (train_x-x_mean)/x_var
    test_x = (test_x-x_mean)/x_var
    train_y = (train_y-y_mean)/y_var
    test_y = (test_y-y_mean)/y_var

    # Create a validation set as a subset of the training set

    val_indices = np.random.permutation(np.arange(len(train_x)))<int(val_prc*len(train_x))
    val_x = train_x[val_indices,:]
    val_y = train_y[val_indices]
    train_x = train_x[~val_indices,:]
    train_y = train_y[~val_indices]

    return train_x, train_y, val_x, val_y, test_x, test_y

def load_boston():

    return None

