import h5py
import numpy as np
import os
import pandas as pd
import pytest

from data_loader import DataLoader
from data import Data


def transform(x):
    return x + 10


def transform_niftii(image):
    return np.flip(image.get_data(), axis=1).copy()


cfg_table = {
    'read_mode': 'table',
    'paths': {
        'table': 'test_table.csv'
    },
    'dataset': {
        'columns': ['f1', 'f2'],
        'label': 'y',
        'stratify_cols': ['f1', 'y'],
        'transform': transform,
        'n_samples': 8,
        'use_atlas': False,
        'minmax_normalize': False
    }
}

cfg_h5 = {
    'read_mode': 'h5',
    'paths': {
        'h5': {
            'train': 'test_h5_train.h5',
            'test': 'test_h5_test.h5',
        }
    },
    'dataset': {
        'transform': transform,
        'use_atlas': False,
        'minmax_normalize': False
    }
}

cfg_h5_tr = {
    'read_mode': 'h5',
    'paths': {
        'h5': {
            'train': 'test_h5.h5',
        }
    },
    'dataset': {
        'transform': transform,
        'use_atlas': False,
        'minmax_normalize': False,
        'use_holdout': False
    }
}

cfg_bids = {
    'read_mode': 'BIDS',
    'paths': {
        'data': './',
        'atlas': '../utils/labels_Neuromorphometrics.nii',
    },
    'dataset': {
        'transform': None,
        'use_atlas': True,
        'minmax_normalize': False,
        'use_holdout': False
    }
}

data = np.array([[1, 1, 1.1, 10],
                 [0, 2, 1.2, 10],
                 [1, 3, 1.3, 10],
                 [0, 4, 1.5, 20],
                 [1, 5, 1.6, 20],
                 [0, 6, 1.6, 20],
                 [1, 7, 1.7, 20],
                 [0, 8, 1.8, 20],
                 [1, 9, 1.9, 20],
                 ])

data_Xtr_h5 = np.array([[1., 3.],
                        [1., 1.],
                        [1., 5.],
                        [0., 8.],
                        [1., 7.],
                        [1., 9.]])

data_Xte_h5 = np.array([[0., 4.],
                        [0., 2.]])

data_ytr_h5 = np.array([5., 5., 10., 10., 10., 10.])

data_yte_h5 = np.array([10., 5.])

# expected values
X_train = np.array([[11., 13.],
                    [11., 11.],
                    [11., 15.],
                    [10., 18.],
                    [11., 17.],
                    [11., 19.]])

X_test = np.array([[10., 14.],
                   [10., 12.]])

y_train = np.array([0., 0., 1., 1., 1., 1.])

y_test = np.array([1., 0.])

expected = Data(X_train, y_train, X_test, y_test)
expected_tr = Data(X_train, y_train)


def compare_data_obj(d1, d2):
    assert np.array_equal(d1.X_train, d2.X_train)
    assert np.array_equal(d1.y_train, d2.y_train)
    if d1.X_test is not None:
        assert np.array_equal(d1.X_test, d2.X_test)
    else:
        assert d2.X_test is None
    if d1.y_test is not None:
        assert np.array_equal(d1.y_test, d2.y_test)
    else:
        assert d2.y_test is None


def test_h5_train_test():
    h_tr = h5py.File('test_h5_train.h5', 'w')
    h_tr.create_dataset('X', data=data_Xtr_h5)
    h_tr.create_dataset('y', data=data_ytr_h5)
    h_tr.close()

    h_te = h5py.File('test_h5_test.h5', 'w')
    h_te.create_dataset('X', data=data_Xte_h5)
    h_te.create_dataset('y', data=data_yte_h5)
    h_te.close()
    try:
        dl = DataLoader(cfg_h5, **cfg_h5['dataset'])
        compare_data_obj(dl.data, expected)
    finally:
        if os.path.exists('test_h5_train.h5'):
            os.remove('test_h5_train.h5')
        if os.path.exists('test_h5_test.h5'):
            os.remove('test_h5_test.h5')


def test_h5_train():
    h = h5py.File('test_h5.h5', 'w')
    h.create_dataset('X', data=data_Xtr_h5)
    h.create_dataset('y', data=data_ytr_h5)
    h.close()
    try:
        dl = DataLoader(cfg_h5_tr, **cfg_h5_tr['dataset'])
        compare_data_obj(dl.data, expected_tr)
    finally:
        if os.path.exists('test_h5.h5'):
            os.remove('test_h5.h5')


def test_table():
    df = pd.DataFrame(data=data, columns=('f1', 'f2', 'f3', 'y'))
    df.to_csv('test_table.csv')
    try:
        dl = DataLoader(cfg_table, **cfg_table['dataset'])
        compare_data_obj(dl.data, expected)
        # TODO test stratification
    finally:
        if os.path.exists('test_table.csv'):
            os.remove('test_table.csv')


@pytest.fixture
def mock_traverse_subj(monkeypatch):
    def mock_get_img_paths(*args, **kwargs):
        return pd.DataFrame({'path': ['../utils/labels_Neuromorphometrics.nii'],
                             'subject': ['sub-01'],
                             'label': [0],
                             'gender': 1
                             })
    monkeypatch.setattr('data_loader._get_img_paths', mock_get_img_paths)


@pytest.mark.usefixtures("mock_traverse_subj")
def test_bids():
    dl = DataLoader(cfg_bids, **cfg_bids['dataset'])
    assert dl.data.X_train.shape == (1, 136), "Atlas did not reduce values properly"
