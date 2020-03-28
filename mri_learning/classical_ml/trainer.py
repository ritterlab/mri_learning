import json
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

from collections import OrderedDict
from joblib import dump
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV


def _extract_estimator_name(e):
    try:
        name = e.__dict__.get('steps')[-1][0]
    except TypeError:
        name = e.__class__.__name__

    return name


def _plot_results(df, scoring, estimator):
    # TODO save figure
    df_plot = df[list(filter(lambda col: col.endswith('_mean_test'), df.columns))]
    df_plot.columns = scoring
    plt.figure()
    estimator_name = _extract_estimator_name(estimator)
    plt.title(f"Validation Scores for {estimator_name}")
    df_plot.boxplot(grid=False, rot=45)
    plt.show()


def _parse_results(cv_results, scoring):
    """
    :param cv_results: dict of numpy (masked) ndarrays. See GridSearchCV
    :param scoring: list or dictionary with scoring functions
    :return: OrderedDict with the results of every scoring function
    """
    dic_results = OrderedDict()
    if isinstance(scoring, list):
        scoring_lst = scoring
    elif isinstance(scoring, dict):
        scoring_lst = scoring.keys()
    else:
        raise TypeError('scoring must be list or dictionary')
    for scorer in scoring_lst:
        # add mean score
        dic = OrderedDict({f"{scorer}_mean_test": cv_results[f"mean_test_{scorer}"].max()})
        best_idx = cv_results[f"rank_test_{scorer}"].argmin()
        best_params = cv_results['params'][best_idx]
        # add values of the highest ranked parameters
        for par, val in best_params.items():
            dic[f"{scorer}_{par}"] = val
        dic_results.update(dic)

    return dic_results


def _save_results(results, models, results_path, extra_suffix=''):
    # save full results
    results_path = os.path.abspath(results_path)
    path = results_path.split('.')
    path[-2] = path[(-2)].replace('config', 'results_full') + extra_suffix
    path[-1] = path[(-1)].replace('json', 'xlsx')
    path = '.'.join(path)

    all_results = sorted((m, r['df']) for m, r in results.items())
    df = pd.concat([df for m, df in all_results], axis=1, keys=[m for m, df in all_results])
    df.to_excel(path)
    # save result summary
    means = pd.DataFrame(dict(mean=(df.mean()), std=(df.std())))
    path = results_path.split('.')
    path[-2] = path[(-2)].replace('config', 'results_agg') + extra_suffix
    path[-1] = path[(-1)].replace('json', 'csv')
    path = '.'.join(path)
    means.to_csv(path)
    # save  models
    path = results_path.split('.')
    path[-2] = path[(-2)].replace('config', 'best_models') + extra_suffix
    path[-1] = path[(-1)].replace('json', 'joblib')
    path = '.'.join(path)
    dump(dict([(m, r['gs']) for m, r in results.items()]), path)
    # save overall
    path = results_path.split('.')
    path[-2] = path[(-2)].replace('config', 'analysis') + extra_suffix
    path[-1] = path[(-1)].replace('json', 'joblib')
    path = '.'.join(path)
    with open(results_path) as f:
        cfg = json.load(f)
    dump(dict(results=results, cfg=cfg), path)


class Trainer:

    def __init__(self,
                 models,
                 parameter_lst,
                 n_splits=3,
                 scoring=('balanced_accuracy',),
                 retrain_metric='balanced_accuracy',
                 trials=5,
                 cfg_path=None,
                 regression=False
                 ):
        self.models = models
        self.parameter_lst = parameter_lst
        self.n_splits = n_splits
        self.scoring = list(scoring) if isinstance(scoring, tuple) else scoring
        self.retrain_metric = retrain_metric
        self.trials = trials
        self.cfg_path = cfg_path
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
        gs = GridSearchCV(model, parameters, scoring=self.scoring, cv=cv, n_jobs=-1,
                          refit=self.retrain_metric)
        gs = gs.fit(X, y)
        total_time = (time.time() - start_time) / 60
        results_data = _parse_results(gs.cv_results_, self.scoring)
        df = pd.DataFrame(data=results_data, index=(0,))
        df['trial'] = trial
        return df.set_index('trial'), gs, total_time

    def run_experiments(self, data, model, parameters, verbose, plotting):
        """
        Runs multiple independent trials for  CV analysis
        and stores the results and best model of each trial
        :param data: Data object
        :param model: sklearn-like estimator
        :param parameters: list of parameters for `model`. See `GridSearchCV`
        :param verbose: verbosity level
        :param plotting: flag to draw a box plot of the aggregated results
        :return: dictionary with the results of every trial
        """
        results = {'df': pd.DataFrame(), 'gs': [], 'times': []}
        if verbose > 0:
            print(f"Starting analysis for model {model} \n with parameters {parameters}")
        for i in range(1, self.trials + 1):
            if verbose > 1:
                print(f"iter: {i}")
            df_results, gs, total_time = self.train_grid_search(data, model, parameters, i)
            results['df'] = pd.concat([results['df'], df_results])
            results['gs'].append(gs)
            results['times'].append(total_time)

        if plotting:  # TODO have more control over this part, possible save images
            try:
                _plot_results(results['df'], self.scoring, model)
            except:
                print("problem while plotting")
        if verbose > 2:
            print(results['df'])
        return results

    def run_models(self, data, verbose=0, plotting=False, extra_suffix=''):
        """
        Repeats a multiple iteration cv analysis for multiple models. See `run_experiments`.
        It saves the results as csv tables for the results, and `joblib` objects for the models
        :param data: Data object
        :param verbose: verbosity level
        :param plotting: flag for plotting
        :param extra_suffix: helper to differentiate between analyses
        :return: pd.DataFrame with the information of each trial,
                 nested list of type [[modelA_1, modelA_2], [modelB_1, modelB_2], ...]
        """
        results = {}
        for model, params in zip(self.models, self.parameter_lst):
            result = self.run_experiments(data, model, params, verbose, plotting)
            results[_extract_estimator_name(model)] = result

        if self.cfg_path:
            _save_results(results, self.models, self.cfg_path, extra_suffix)
        return results
