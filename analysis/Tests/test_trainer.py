from trainer import Trainer

import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline


class Data:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

#TODO create cfg file in the fly

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

t1 = Trainer([m1], [params1], trials=1, scoring=[scoring1])
results1 = t1.run_models(d, 0)

t2 = Trainer([m1], [params1], trials=3, scoring=[scoring1])
results2 = t2.run_models(d, 0)

t3 = Trainer([m1, m2], [params1, params2], trials=1, scoring=[scoring1, scoring2])
results3 = t3.run_models(d, 0)

t4 = Trainer([m1, m2], [params1, params2], trials=3, scoring=[scoring1, scoring2],
                     cfg_path='./classical_ml/mock_results/config.json')
results4 = t4.run_models(d, 0)


def check_estimators(results, estimators):
    assert sorted(results.keys()) == estimators, 'Mismatch between results and configuration estimators'


def check_length(results, trials):
    for e, result in results.items():
        for i, val in result.items():
            assert len(val) == trials, f'Length of results {e, i} != than {trials}'


def mean_metric(results, metric, expected_scores):
    for m in expected_scores:
        score = results[m]['df'][f'{metric}_mean_test'].mean()
        true_score = expected_scores[m]
        assert np.isclose(score, true_score, 1e-2), f'Unexpected value for {metric} and {m}, {score} != {true_score}'


def test_simple_gs_cv():
    check_estimators(results1, [e1])
    check_length(results1, 1)
    mean_metric(results1, scoring1, {e1: 0.5})


def test_trials_gs_cv():
    check_estimators(results2, [e1])
    check_length(results2, 3)
    mean_metric(results2, scoring1, {e1: 0.5})


def test_models_metrics_gs_cv():
    check_estimators(results3, [e1, e2])
    check_length(results3, 1)
    mean_metric(results3, scoring1, {e1: 0.5, e2: 1})


def test_models_trials_metrics_gs_cv():
    check_estimators(results4, [e1, e2])
    check_length(results4, 3)
    mean_metric(results4, scoring1, {e1: 0.5, e2: 1})
    mean_metric(results4, scoring2, {e1: 1, e2: 1})

