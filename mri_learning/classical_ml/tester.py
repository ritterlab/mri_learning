import argparse
import os
import random
import seaborn as sns
import warnings

import pandas as pd
import numpy as np

from configuration import Configuration
from data_loader import DataLoader
from helpers import rename_file, specificity
from joblib import load as jb_load
from matplotlib import pyplot as plt
from scipy.stats import mode


def _save_results(df, results_path, extra_suffix=''):
    """Save a data frame in an excel file"""
    results_path = os.path.abspath(results_path)
    path = rename_file(results_path, 'config', 'results_holdout', 'json', 'xlsx', extra_suffix)
    df.to_excel(path)


class Tester:
    def __init__(self,
                 results=None,
                 cfg_path=None,
                 best_metric='balanced_accuracy',
                 scorers=('balanced_accuracy',),
                 model_selection='all_models',
                 voting=True,
                 plotting=False,
                 chosen_estimator=None,
                 chosen_trial=None
                 ):
        self.cfg_path = os.path.abspath(cfg_path)
        self.results = results
        self.load_results()
        self.best_metric = best_metric
        self.scorers = list(scorers)
        self.model_selection = model_selection
        self.voting = voting
        self.plotting = plotting
        self.chosen_estimator = chosen_estimator
        self.chosen_trial = chosen_trial
        self.n_trials = self._get_n_trials()
        self.chosen_models = None
        self.predictions = None
        self.extra_suffix = ''

    def _get_n_trials(self):
        return len(self.results[next(iter(self.results.keys()))]['gs'])

    def _get_model_selection(self):
        methods = {
            'all_models': self.all_models,
            'best_model': self.best_model,
            'best_model_type': self.best_estimator_type,
            'best_model_type_random': self.best_estimator_random,
            'random_model': self.random_model,
            'random_model_all': self.random_all_types,
            'specific_model': self.specific_model
        }
        return methods[self.model_selection]

    def load_results(self):
        if self.results is None:
            path = rename_file(self.cfg_path, 'config', 'mri_learning', 'json', 'joblib')
            self.results = jb_load(path)['results']

    def load_data(self):
        cfg = Configuration(self.cfg_path).cfg
        dl = DataLoader(cfg, **cfg['dataset'])
        return dl.data.X_test, dl.data.y_test

    def run(self, X_test=None, y_test=None):
        """
        Evaluates the performance of the selected model on X_test, y_test
        :param X_test: np.array
        :param y_test: np.array
        :return: pd.DataFrame
        """
        if X_test is None or y_test is None:
            X_test, y_test = self.load_data()
        select_method = self._get_model_selection()
        self.chosen_models = select_method()
        self.predictions = self.predict(X_test)
        return self.score(y_test, self.predictions)

    def all_models(self):
        """return models from all estimator types and trials"""
        self.chosen_estimator = None
        self.chosen_trial = None
        return dict([(m, r['gs']) for m, r in self.results.items()])

    def best_model(self):
        """return overall best model"""
        estimators = sorted(self.results.keys())
        trial, best_estimator = sorted([self.results[e]['df'][f'{self.best_metric}_mean_test'].agg(
            ['max', 'idxmax']).tolist() + [e] for e in estimators], reverse=True)[0][1:]

        self.chosen_estimator = best_estimator
        self.chosen_trial = trial
        return {best_estimator: [self.results[best_estimator]['gs'][int(trial)]]}

    def _get_best_estimator(self):
        estimators = sorted(self.results.keys())
        best_idx = np.array((self.results[e]['df']['balanced_accuracy_mean_test'].mean() for e in estimators)).argmax()
        return estimators[best_idx]

    def best_estimator_type(self):
        """return models from the estimator that performed better"""
        best_estimator = self._get_best_estimator()
        self.chosen_estimator = best_estimator
        self.chosen_trial = None
        return {best_estimator: self.results[best_estimator]['gs']}

    def best_estimator_random(self):
        """return a random model from the estimator that performed better"""
        random.seed(0)
        best_estimator = self._get_best_estimator()
        chosen_trial = random.randint(0, self.n_trials - 1)
        self.chosen_estimator = best_estimator
        self.chosen_trial = chosen_trial
        return {best_estimator: [random.choice(self.results[best_estimator]['gs'])]}

    def random_model(self):
        """return a random model"""
        random.seed(0)
        e, models = random.choice(list(self.results.items()))
        chosen_trial = random.randint(0, self.n_trials - 1)
        self.chosen_estimator = e
        self.chosen_trial = chosen_trial
        return {e: [models['gs'][chosen_trial]]}

    def random_all_types(self):
        """return models from all estimator types from a random trial"""
        random.seed(0)
        chosen_trial = random.randint(0, self.n_trials - 1)
        self.chosen_estimator = None
        self.chosen_trial = chosen_trial
        return dict([(e, [self.results[e]['gs'][chosen_trial]]) for e in self.results.keys()])

    def specific_model(self):
        """return a model from the specified estimator kind and trial"""
        return {self.chosen_estimator: [self.results[self.chosen_estimator]['gs'][self.chosen_trial]]}

    def predict(self, X_test):
        """
        Create prediction vectors for every stored model. If voting is enabled,
        aggregates results of selected classifiers
        :param X_test:
        :return: dictionary of the form {estimator_name: [np.array, ...], ...}
        """
        predictions = dict([(e, [m.predict(X_test) for m in ms]) for e, ms in self.chosen_models.items()])
        if self.voting:
            predictions['VotingClassifier'] = [mode([p[t] for p in predictions.values()]
                                                    )[0][0] for t in range(self.n_trials)]
        return predictions

    def plot_results(self, df):
        output_image = rename_file(self.cfg_path, 'config', 'results_holdout', 'json', 'png')
        plt.figure(figsize=(12, 8))
        plot_df = pd.concat([df[['estimator', s]].rename(columns={s: 'score'}).assign(metric=s)
                             for s in [s.__name__ for s in self.scorers]])
        sns_plot = sns.barplot(x='estimator', y='score', hue='metric', data=plot_df)
        sns_plot.get_figure().savefig(output_image)

    def score(self, y_test, predictions):
        """
        This function evaluates predictions of models according to the defined scoring functions
        :param y_test: 1-d np.array-like
        :param predictions: dictionary, see tester.predict
        :return: pd.DataFrame with results per metric, model, and trial
        """
        all_metrics = []
        estimators = predictions.keys()
        for e in sorted(estimators):
            all_metrics.append(pd.DataFrame([[s(p, y_test) for s in self.scorers] for p in predictions[e]],
                                            columns=[s.__name__ for s in self.scorers]))
        df = pd.concat(all_metrics, keys=estimators).reset_index(
                ).drop(columns='level_1').rename(columns={'level_0': 'estimator'})
        df_agg = df.groupby(['estimator']).agg(['mean', 'std'])
        if self.plotting:
            self.plot_results(df)
        if self.cfg_path:
            _save_results(df, self.cfg_path, self.extra_suffix)
        print(df_agg)  # TODO should be dumped?
        return df


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="path to the configuration file")
    path = parser.parse_args().cfg_path
    cfg = Configuration(path).cfg
    dataset = DataLoader(cfg, **cfg['dataset'])
    tester = Tester(**cfg['holdout'])
    results = tester.run(dataset.data.X_test, dataset.data.y_test)

