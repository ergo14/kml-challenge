from pathlib import Path

import numpy as np
import pandas as pd


def scale_features(arr, scale_method):
    if scale_method == 'minmax':
        rng = arr.max(axis=0) - arr.min(axis=0)
        rng[np.isclose(rng, 0)] = 1
        return (arr - arr.min(axis=0)) / rng
    elif scale_method == 'std':
        rng = arr.std(axis=0)
        rng[np.isclose(rng, 0)] = 1
        return (arr - arr.mean(axis=0)) / rng
    else:
        raise ValueError('Unrecognized scaling method. Choose among minmax and std.')


def get_text_data(dataset, path):
    path = Path(path)
    train_features = pd.read_csv(path.joinpath(f'Xtr{dataset}.csv'), sep=',', index_col='Id')
    train_response = pd.read_csv(path.joinpath(f'Ytr{dataset}.csv'), sep=',', index_col='Id', header=0)
    prediction_features = pd.read_csv(path.joinpath(f'Xte{dataset}.csv'), sep=',', index_col='Id')
    return train_features, train_response, prediction_features


def get_numeric_data(dataset, path):
    train_features = pd.read_csv(Path(path).joinpath(f'Xtr{dataset}_mat100.csv'), sep=' ', header=None)
    train_response = pd.read_csv(Path(path).joinpath(f'Ytr{dataset}.csv'), sep=',', index_col='Id', header=0)
    prediction_features = pd.read_csv(Path(path).joinpath(f'Xte{dataset}_mat100.csv'), sep=' ', header=None)
    return train_features, train_response, prediction_features


def split_train_test(X, y, num_train):
    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]
    return X_train, y_train, X_test, y_test


def prepare_numeric_data(dataset, path, scale_method, with_intercept, num_train):
    df_tr, df_y, df_te = get_numeric_data(dataset=dataset, path=path)
    X, y, X_te = df_tr.values, df_y.values, df_te.values

    if with_intercept:
        X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
        X_te = np.concatenate([np.ones([X_te.shape[0], 1]), X_te], axis=1)
    
    if scale_method is not None:
        X = scale_features(X, scale_method=scale_method)
        X_te = scale_features(X_te, scale_method=scale_method)

    X_train, y_train, X_val, y_val = split_train_test(X, y, num_train=num_train)

    return X_train, y_train, X_val, y_val, X_te


def compute_accuracy(model, X, y):
    prd = model.predict_class(X)
    return (prd == y).mean()















