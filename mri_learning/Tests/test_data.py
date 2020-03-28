import numpy as np
from copy import deepcopy
from data import Data

# data
X_tr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=float)
y_tr = np.array([0, 1, 1, 0, 1, 1], dtype=float)
X_te = np.array([[9, 1, 3], [8, 4, 0]], dtype=float)
y_te = np.array([0, 1], dtype=float)
X_ex = np.array([[100], [200], [300], [400], [500], [600]], dtype=float)
y_ex = np.array([0, 1, 1, 0, 1, 1], dtype=float)

# expected_outputs
merged = Data(np.array([[1, 2, 3, 100], [4, 5, 6, 200], [7, 8, 9, 300],
                        [0, 1, 2, 400], [3, 4, 5, 500], [6, 7, 8, 600]], dtype=float),
              np.array([0, 1, 1, 0, 1, 1], dtype=float))
data_mn = Data(np.array([[-1, -1, -1],
                         [0.2, 0.2, 0.2],
                         [1.4, 1.4, 1.4],
                         [-1.4, -1.4, -1.4],
                         [-0.2, -0.2, -0.2],
                         [1, 1, 1]], dtype=float),
               np.array([0, 1, 1, 0, 1, 1], dtype=float),
               np.array([[2.2, -1.4, -1],
                         [1.8, -0.2, -2.2]], dtype=float),
               np.array([0, 1], dtype=float))

data_mm = Data(np.array([[0., 0.5, 1.],
                         [0., 0.5, 1.],
                         [0., 0.5, 1.],
                         [0., 0.5, 1.],
                         [0., 0.5, 1.],
                         [0., 0.5, 1.]], dtype=float),
               np.array([0, 1, 1, 0, 1, 1], dtype=float),
               np.array([[1., 0., 0.25],
                         [1., 0.5, 0.]], dtype=float),
               np.array([0, 1], dtype=float))


def compare(d1, d2):
    def all_close(x, y):
        if x is not None and y is not None:
            return np.allclose(x, y)
        elif x is None and y is None:
            return True
        else:
            return False

    assert len(d1) == len(d2)
    assert all_close(d1.X_train, d2.X_train)
    assert all_close(d1.y_train, d2.y_train)
    assert all_close(d1.X_test, d2.X_test)
    assert all_close(d1.y_test, d2.y_test)


def merge(d1, d2, d_ex):
    print("Testing merge method")
    d3 = d1.merge(d2)
    compare(d3, d_ex)


def minmax_normalization(d_in, d_ex):
    print("Test min max normalization")
    d_in.minmax_normalize()
    compare(d_in, d_ex)


def mean_normalize(d_in, d_ex):
    print("Test standarization")
    d_in.mean_normalize()
    compare(d_in, d_ex)


def test_data_class():
    data_all = Data(X_tr, y_tr, X_te, y_te)
    data_trn = Data(X_tr, y_tr)
    data_ext = Data(X_ex, y_ex)
    # test on Data with training and test set
    merge(data_trn, data_ext, merged)
    minmax_normalization(deepcopy(data_all), data_mm)
    mean_normalize(deepcopy(data_all), data_mn)
    # test on Data with training set only
    data_mm.X_test, data_mm.y_test = None, None
    data_mn.X_test, data_mn.y_test = None, None
    minmax_normalization(deepcopy(data_trn), data_mm)
    mean_normalize(deepcopy(data_trn), data_mn)