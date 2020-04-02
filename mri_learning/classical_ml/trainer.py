import json
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

from collections import OrderedDict
from joblib import dump
from helpers import rename_file
from helpers import specificity
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline


def _extract_estimator_name(e):
    try:
        name = e.__dict__.get('steps')[-1][0]
    except TypeError:
        name = e.__class__.__name__

    return name


def _plot_results(df, scorers, estimator):
    # TODO save figure
    df_plot = df[list(filter(lambda col: col.endswith('_mean_test'), df.columns))]
    df_plot.columns = scorers
    plt.figure()
    estimator_name = _extract_estimator_name(estimator)
    plt.title(f"Validation Scores for {estimator_name}")
    df_plot.boxplot(grid=False, rot=45)
    plt.show()


def _parse_results(cv_results, scorers, train_scores=False):
    """
    :param cv_results: dict of numpy (masked) ndarrays. See GridSearchCV
    :param scorers: list or dictionary with scorers functions
    :param train_scores: bool: flag that determines if train scores should be added
    :return: OrderedDict with the results of every scorer function
    """
    dic_results = OrderedDict()
    if isinstance(scorers, list):
        scorers_lst = scorers
    elif isinstance(scorers, dict):
        scorers_lst = scorers.keys()
    else:
        raise TypeError('scorers must be list or dictionary')
    for scorer in scorers_lst:
        # add mean score
        best_idx = cv_results[f"rank_test_{scorer}"].argmin()
        dic = OrderedDict({f"{scorer}_mean_test": cv_results[f"mean_test_{scorer}"][best_idx]})
        if train_scores:
            dic.update({f"{scorer}_mean_train": cv_results[f"mean_train_{scorer}"][best_idx]})
        best_params = cv_results['params'][best_idx]
        # add values of the highest ranked parameters
        for par, val in best_params.items():
            dic[f"{scorer}_{par}"] = val
        dic_results.update(dic)

    return dic_results


def _save_results(results, models, results_path, extra_suffix=''):
    # save full results
    results_path = os.path.abspath(results_path)
    path = rename_file(results_path, 'config', 'results_full', 'json', 'xlsx', extra_suffix)

    all_results = sorted((m, r['df']) for m, r in results.items())
    df = pd.concat([df for m, df in all_results], axis=1, keys=[m for m, df in all_results])
    df.to_excel(path)
    # save result summary
    means = pd.DataFrame(dict(mean=(df.mean()), std=(df.std())))
    path = rename_file(results_path, 'config', 'results_agg', 'json', 'csv', extra_suffix)
    means.to_csv(path)
    # save  models
    path = rename_file(results_path, 'config', 'best_models', 'json', 'joblib', extra_suffix)
    dump(dict([(m, r['gs']) for m, r in results.items()]), path)
    # save overall
    path = rename_file(results_path, 'config', 'mri_learning', 'json', 'joblib', extra_suffix)
    with open(results_path) as f:
        cfg = json.load(f)
    dump(dict(results=results, cfg=cfg), path)


class Trainer:

    def __init__(self,
                 models=None,
                 parameter_lst=None,
                 n_splits=3,
                 scorers=('balanced_accuracy',),
                 retrain_metric='balanced_accuracy',
                 return_train_score=False,
                 trials=5,
                 cfg_path=None,
                 verbose=0,
                 plotting=False,
                 extra_suffix='',
                 regression=False
                 ):
        self.models = models if models else [Pipeline([('DummyClassifier', DummyClassifier())])]
        self.parameter_lst = parameter_lst if parameter_lst else [{'DummyClassifier__strategy': ['most_frequent']}]
        self.n_splits = n_splits
        self.scorers = list(scorers) if isinstance(scorers, tuple) else scorers
        self.retrain_metric = retrain_metric
        self.return_train_score = return_train_score
        self.trials = trials
        self.cfg_path = cfg_path
        self.verbose = verbose
        self.plotting = plotting
        self.extra_suffix = extra_suffix
        self.regression = regression
        self.KFoldStrategy = KFold if regression else StratifiedKFold

    def train_grid_search(self, data, model, parameters, trial=1):
        """
        Trains a dataset with a specified model and parameter grid
        :param data: Data Object with X_train, y_train attributes
        :param model: sklearn-like estimator
        :param parameters: list of parameters for `model`. See `GridSearchCV`
        :param trial: Experiment iteration number
        :return: pd.DataFrame with the summarized fold information,
                 grid search estimator refit with all training set and best parameters according to `retrain_metric`,
                 training time
        """
        start_time = time.time()
        X, y = data.X_train, data.y_train
        cv = tuple(self.KFoldStrategy(n_splits=self.n_splits, shuffle=True, random_state=trial).split(X, y))
        gs = GridSearchCV(model, parameters, scoring=self.scorers, cv=cv, n_jobs=-1,
                          refit=self.retrain_metric, return_train_score=self.return_train_score)
        gs = gs.fit(X, y)
        total_time = (time.time() - start_time) / 60
        results_data = _parse_results(gs.cv_results_, self.scorers, self.return_train_score)
        df = pd.DataFrame(data=results_data, index=(0,))
        df['trial'] = trial
        return df.set_index('trial'), gs, total_time

    def run_experiments(self, data, model, parameters):
        """
        Runs multiple independent trials for  CV analysis
        and stores the results and best model of each trial
        :param data: Data object
        :param model: sklearn-like estimator
        :param parameters: list of parameters for `model`. See `GridSearchCV`
        :return: dictionary with the results of every trial
        """
        results = {'df': pd.DataFrame(), 'gs': [], 'times': []}
        if self.verbose > 0:
            print(f"Starting analysis for model {model} \n with parameters {parameters}")
        for i in range(1, self.trials + 1):
            if self.verbose > 1:
                print(f"iter: {i}")
            df_results, gs, total_time = self.train_grid_search(data, model, parameters, i)
            results['df'] = pd.concat([results['df'], df_results])
            results['gs'].append(gs)
            results['times'].append(total_time)

        if self.plotting:  # TODO have more control over this part, possible save images
            try:
                _plot_results(results['df'], self.scorers, model)
            except:
                print("problem while plotting")
        if self.verbose > 2:
            print(results['df'])
        return results

    def run_models(self, data):
        """
        Repeats a multiple iteration cv analysis for multiple models. See `run_experiments`.
        It saves the results as csv tables for the results, and `joblib` objects for the models
        :param data: Data object
        :return: pd.DataFrame with the information of each trial,
                 nested list of type [[modelA_1, modelA_2], [modelB_1, modelB_2], ...]
        """
        results = {}
        for model, params in zip(self.models, self.parameter_lst):
            result = self.run_experiments(data, model, params)
            results[_extract_estimator_name(model)] = result

        if self.cfg_path:
            _save_results(results, self.models, self.cfg_path, self.extra_suffix)
        return results
