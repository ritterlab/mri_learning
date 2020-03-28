import h5py
import numpy as np
import os
import pandas as pd
import pytest

from data_loader import DataLoader


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


def test_h5_train_test(h5_data, cfg_h5, expected):
    h_tr = h5py.File('test_h5_train.h5', 'w')
    h_tr.create_dataset('X', data=h5_data['X_train'])
    h_tr.create_dataset('y', data=h5_data['y_train'])
    h_tr.close()

    h_te = h5py.File('test_h5_test.h5', 'w')
    h_te.create_dataset('X', data=h5_data['X_test'])
    h_te.create_dataset('y', data=h5_data['y_test'])
    h_te.close()
    try:
        dl = DataLoader(cfg_h5, **cfg_h5['dataset'])
        compare_data_obj(dl.data, expected)
    finally:
        if os.path.exists('test_h5_train.h5'):
            os.remove('test_h5_train.h5')
        if os.path.exists('test_h5_test.h5'):
            os.remove('test_h5_test.h5')


def test_h5_train(h5_data, cfg_h5_tr, expected_tr):
    h = h5py.File('test_h5.h5', 'w')
    h.create_dataset('X', data=h5_data['X_train'])
    h.create_dataset('y', data=h5_data['y_train'])
    h.close()
    try:
        dl = DataLoader(cfg_h5_tr, **cfg_h5_tr['dataset'])
        compare_data_obj(dl.data, expected_tr)
    finally:
        if os.path.exists('test_h5.h5'):
            os.remove('test_h5.h5')


def test_table(data, cfg_table, expected):
    df = pd.DataFrame(data=data, columns=('f1', 'f2', 'f3', 'y'))
    df.to_csv('test_table.csv')
    try:
        dl = DataLoader(cfg_table, **cfg_table['dataset'])
        compare_data_obj(dl.data, expected)
        # TODO test stratification
    finally:
        if os.path.exists('test_table.csv'):
            os.remove('test_table.csv')


@pytest.mark.usefixtures("mock_traverse_subj")
def test_bids(cfg_bids):
    dl = DataLoader(cfg_bids, **cfg_bids['dataset'])
    assert dl.data.X_train.shape == (1, 136), "Atlas did not reduce values properly"
