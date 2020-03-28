import numpy as np
import pytest


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


@pytest.mark.parametrize("test_kind, trials, means",
                         [('simple', 1, [[0.5]]),
                          ('multi_trial', 3, [[0.5]]),
                          ('multi_model', 1, [[0.5, 1]]),
                          ('complete', 3, [[0.5, 1], [1, 1]])
                          ])
def test_gs_cv(test_kind, trials, means, all_results, all_estimators, all_scorers):
    results = all_results[test_kind]
    estimators = all_estimators[test_kind]
    scoring = all_scorers[test_kind]

    check_estimators(results, estimators)
    check_length(results, trials)
    for s, m in zip(scoring, means):
        mean_metric(results, s, dict(zip(estimators, m)))
