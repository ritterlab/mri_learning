import argparse
import json
import warnings

from data_loader import DataLoader
from numpy.random import seed as np_seed
from random import seed
from importlib import import_module
from sklearn.metrics import make_scorer, SCORERS
from sklearn.pipeline import Pipeline
from trainer import Trainer


def isinstance_none(obj, data_type):
    return isinstance_none(obj, data_type) or obj is None


def _get_model(model_name):
    try:
        m = eval(model_name)
    except NameError:
        sklearn_models = {
            'LogisticRegression': 'sklearn.linear_model',
            'SVC': 'sklearn.svm',
            'GradientBoostingClassifier': 'sklearn.ensemble',
            'PCA': 'sklearn.decomposition'
        }
        m = getattr(import_module(sklearn_models[model_name]), model_name)
    return m()


class Configuration:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.cfg = self.read()
        self.type_check()
        self.check_cfg()
        self.parse_cfg()

    def read(self):
        with open(self.cfg_path, 'r') as f:
            cfg = json.load(f)
        return cfg

    def type_check(self):
        assert isinstance(self.cfg['read_mode'], str)
        assert isinstance(self.cfg['paths'], dict)
        assert isinstance(self.cfg['dataset'], dict)
        assert isinstance(self.cfg['training'], dict)

        dataset = self.cfg['dataset']
        assert isinstance_none(dataset.get('n_samples'), int)
        assert isinstance_none(float(dataset.get('test_size')), float)
        assert isinstance_none(dataset.get('label'), str)
        assert isinstance_none(dataset.get('atlas_strategy'), str)
        assert isinstance_none(dataset.get('use_holdout'), bool)
        assert isinstance_none(dataset.get('use_atlas'), bool)
        assert isinstance_none(dataset.get('minmax_normalize'), bool)
        assert isinstance_none(dataset.get('mean_normalize'), bool)
        assert isinstance_none(dataset.get('stratify_columns'), list)
        assert isinstance_none(dataset.get('columns'), list)
        assert isinstance_none(dataset.get('transform'), list)
        assert dataset.get('target_transform') is None or callable(eval(dataset.get('target_transform')))

        training = self.cfg['training']
        assert isinstance_none(training['n_splits'], int)
        assert isinstance_none(training['trials'], int)
        assert isinstance_none(training['verbose'], int)
        assert isinstance_none(training['plotting'], bool)
        assert isinstance_none(training['regression'], bool)
        assert isinstance_none(training['retrain_metric'], str)
        assert isinstance_none(training['scoring'], str) or isinstance_none(training['scoring'], list)
        assert isinstance_none(training['parameter_lst'], list)
        assert isinstance_none(training['models'], list)

    def check_cfg(self):
        read_mode = self.cfg['read_mode']
        assert read_mode in ('h5', 'table', 'BIDS')
        if read_mode == 'h5':
            assert 'train' in self.cfg['paths']['h5']
        elif read_mode == 'table':
            assert 'table' in self.cfg['paths']
        elif read_mode == 'BIDS':
            assert 'data' in self.cfg['paths']

        dataset = self.cfg['dataset']
        if 'n_splits' in dataset:
            assert dataset['n_splits'] > 0
        if 'stratify_columns' in dataset and dataset.get('columns') is not None:
            for sc in dataset['stratify_columns']:
                assert sc in dataset['columns']
        if dataset.get('use_atlas') in (None, True):
            assert 'atlas' in self.cfg['paths']
            assert dataset.get('atlas_strategy') in ('mean', 'sum', None)
        if dataset.get('transforms'):
            for t in dataset.get('transforms'):
                assert callable(eval(t))

    def parse_cfg(self):
        transformations = self.cfg['dataset'].get('transform')
        if transformations:
            self.cfg['dataset']['transform'] = [eval(t) for t in transformations]

        target_transform = self.cfg['dataset'].get('target_transform')
        if target_transform:
            self.cfg['dataset']['target_transform'] = eval(target_transform)

        scoring = self.cfg['training'].get('scoring')
        if isinstance(scoring, str):
            scoring = [scoring]
        scoring_dict = {}
        for s in scoring:
            scoring_dict[s] = s if s in SCORERS.keys() else make_scorer(eval(s))
        self.cfg['training']['scoring'] = scoring_dict

        models = self.cfg['training'].get('models')
        if models:
            self.cfg['training']['models'] = [Pipeline([(e, _get_model(e)) for e in model]) for model in models]

        self.cfg['training']['cfg_path'] = self.cfg_path


class ClassicalML:
    def __init__(self, cfg_path):
        np_seed(0)
        seed(0)
        self.cfg_path = cfg_path
        cfg = Configuration(cfg_path).cfg
        dataset = DataLoader(cfg, **cfg['dataset'])
        t = Trainer(**cfg['training'])
        self.results = t.run_models(dataset.data)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="path to the configuration file")
    path = parser.parse_args().cfg_path
    cml = ClassicalML(path)
    results = cml.results

