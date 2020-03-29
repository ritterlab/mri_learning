import numpy as np
import pandas as pd
import pytest

from data import Data
from trainer import Trainer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline


# data loader
def transform(x):
    return x + 10


def transform1(x):
    return x * 2


def transform_niftii(image):
    return np.flip(image.get_data(), axis=1).copy()


_cfg_table = {
    'read_mode': 'table',
    'paths': {
        'table': 'test_table.csv'
    },
    'dataset': {
        'columns': ['f1', 'f2'],
        'label': 'y',
        'stratify_cols': ['f1', 'y'],
        'transform': [transform],
        'n_samples': 8,
        'use_atlas': False,
        'minmax_normalize': False
    }
}

_cfg_h5 = {
    'read_mode': 'h5',
    'paths': {
        'h5': {
            'train': 'test_h5_train.h5',
            'test': 'test_h5_test.h5',
        }
    },
    'dataset': {
        'transform': [transform],
        'use_atlas': False,
        'minmax_normalize': False
    }
}

_cfg_h5_tr = {
    'read_mode': 'h5',
    'paths': {
        'h5': {
            'train': 'test_h5.h5',
        }
    },
    'dataset': {
        'transform': [transform, transform1],
        'use_atlas': False,
        'minmax_normalize': False,
        'use_holdout': False
    }
}

_cfg_bids = {
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

_data = np.array([[1, 1, 1.1, 10],
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

_h5_data = {'X_train': data_Xtr_h5, 'y_train': data_ytr_h5, 'X_test': data_Xte_h5, 'y_test': data_yte_h5}

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

_expected = Data(X_train, y_train, X_test, y_test)
_expected_tr = Data(X_train * 2, y_train)

# trainer
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
X = np.random.random(size=(10, 5))
X[:, -1] = y
d = Data(X, y)

e1 = 'DummyClassifier'
m1 = Pipeline([(e1, DummyClassifier())])
params1 = {'DummyClassifier__strategy': ['most_frequent', 'constant'], 'DummyClassifier__constant': [0]}

e2_0, e2 = 'SelectKBest', 'RidgeClassifier'
m2 = Pipeline([(e2_0, SelectKBest()), (e2, RidgeClassifier())])
params2 = {'SelectKBest__k': [1]}

scoring1, scoring2 = 'balanced_accuracy', 'recall'

t1 = Trainer(models=[m1], parameter_lst=[params1], trials=1, scorers=[scoring1])
results1 = t1.run_models(d)

t2 = Trainer(models=[m1], parameter_lst=[params1], trials=3, scorers=[scoring1])
results2 = t2.run_models(d)

t3 = Trainer(models=[m1, m2], parameter_lst=[params1, params2], trials=1, scorers=[scoring1, scoring2])
results3 = t3.run_models(d)

t4 = Trainer(models=[m1, m2], parameter_lst=[params1, params2], trials=3, scorers=[scoring1, scoring2],
             cfg_path='./classical_ml/mock_results/config.json')
results4 = t4.run_models(d)


# define fixtures

@pytest.fixture(scope="module")
def cfg_bids():
    return _cfg_bids


@pytest.fixture(scope="module")
def cfg_table():
    return _cfg_table


@pytest.fixture(scope="module")
def cfg_h5():
    return _cfg_h5


@pytest.fixture(scope="module")
def cfg_h5_tr():
    return _cfg_h5_tr


@pytest.fixture(scope="module")
def data():
    return _data


@pytest.fixture(scope="module")
def h5_data():
    return _h5_data


@pytest.fixture(scope="module")
def expected():
    return _expected


@pytest.fixture(scope="module")
def expected_tr():
    return _expected_tr


@pytest.fixture
def mock_traverse_subj(monkeypatch):
    def mock_get_img_paths(*args, **kwargs):
        return pd.DataFrame({'path': ['../utils/labels_Neuromorphometrics.nii'],
                             'subject': ['sub-01'],
                             'label': [0],
                             'gender': 1
                             })

    monkeypatch.setattr('data_loader._get_img_paths', mock_get_img_paths)


@pytest.fixture(scope="module")
def all_results():
    return {'simple': results1,
            'multi_trial': results2,
            'multi_model': results3,
            'complete': results4
            }

@pytest.fixture(scope="module")
def all_scorers():
    return {'simple': [scoring1],
            'multi_trial': [scoring1],
            'multi_model': [scoring1],
            'complete': [scoring1, scoring2]
            }


@pytest.fixture(scope="module")
def all_estimators():
    return {'simple': [e1],
            'multi_trial': [e1],
            'multi_model': [e1, e2],
            'complete': [e1, e2]
            }
