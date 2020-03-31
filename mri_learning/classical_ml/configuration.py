import json

from helpers import isinstance_none, specificity
from importlib import import_module
from sklearn.metrics import make_scorer, SCORERS, balanced_accuracy_score, accuracy_score, roc_auc_score, recall_score
from sklearn.pipeline import Pipeline


def _get_model(model_name):
    try:
        m = eval(model_name)
    except NameError:
        sklearn_models = {
            'LogisticRegression': 'sklearn.linear_model',
            'SVC': 'sklearn.svm',
            'GradientBoostingClassifier': 'sklearn.ensemble',
            'PCA': 'sklearn.decomposition',
            'SelectKBest': 'sklearn.feature_selection'
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
        assert isinstance_none(dataset.get('test_size'), int) or isinstance_none(dataset.get('test_size'), float)
        assert isinstance_none(dataset.get('label'), str)
        assert isinstance_none(dataset.get('atlas_strategy'), str)
        assert isinstance_none(dataset.get('use_holdout'), bool)
        assert isinstance_none(dataset.get('use_atlas'), bool)
        assert isinstance_none(dataset.get('minmax_normalize'), bool)
        assert isinstance_none(dataset.get('mean_normalize'), bool)
        assert isinstance_none(dataset.get('stratify_cols'), list)
        assert isinstance_none(dataset.get('columns'), list)
        assert isinstance_none(dataset.get('transform'), list)
        assert dataset.get('target_transform') is None or callable(eval(dataset.get('target_transform')))

        training = self.cfg['training']
        assert isinstance_none(training.get('n_splits'), int)
        assert isinstance_none(training.get('trials'), int)
        assert isinstance_none(training.get('verbose'), int)
        assert isinstance_none(training.get('plotting'), bool)
        assert isinstance_none(training.get('regression'), bool)
        assert isinstance_none(training.get('retrain_metric'), str)
        assert isinstance_none(training.get('scorers'), str) or isinstance_none(training.get('scorers'), list)
        assert isinstance_none(training.get('parameter_lst'), list)
        assert isinstance_none(training.get('models'), list)

        holdout = self.cfg.get('holdout')
        if holdout:
            assert isinstance_none(holdout.get('retrain_metric'), str)
            assert isinstance_none(holdout.get('model_selection'), str)
            assert isinstance_none(holdout.get('scorers'), list)
            assert isinstance_none(holdout.get('plotting'), bool)
            assert isinstance_none(holdout.get('voting'), bool)
            assert isinstance_none(holdout.get('chosen_trial'), int)
            assert isinstance_none(holdout.get('chosen_estimator'), str)

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
        if 'stratify_cols' in dataset and dataset.get('columns') is not None:
            for sc in dataset['stratify_cols']:
                assert sc in dataset['columns']
        if dataset.get('use_atlas') in (None, True):
            assert 'atlas' in self.cfg['paths']
            assert dataset.get('atlas_strategy') in ('mean', 'sum', None)
        if dataset.get('transforms'):
            for t in dataset.get('transforms'):
                assert callable(eval(t))
        holdout = self.cfg.get('holdout')
        if holdout:
            if 'model_selection' in holdout:
                assert holdout['model_selection'] in ('all_models',
                                                      'best_model',
                                                      'best_model_type',
                                                      'best_model_type_random',
                                                      'random_model',
                                                      'random_model_all',
                                                      'specific_model')
                if holdout['model_selection'] != 'all_models':
                    assert not holdout.get('voting')

    def parse_cfg(self):
        transformations = self.cfg['dataset'].get('transform')
        if transformations:
            self.cfg['dataset']['transform'] = [eval(t) for t in transformations]

        target_transform = self.cfg['dataset'].get('target_transform')
        if target_transform:
            self.cfg['dataset']['target_transform'] = eval(target_transform)

        scorers = self.cfg['training'].get('scorers')
        if isinstance(scorers, str):
            scorers = [scorers]
        scorers_dict = {}
        for s in scorers:
            scorers_dict[s] = s if s in SCORERS.keys() else make_scorer(eval(s))
        self.cfg['training']['scorers'] = scorers_dict

        models = self.cfg['training'].get('models')
        if models:
            self.cfg['training']['models'] = [Pipeline([(e, _get_model(e)) for e in model]) for model in models]

        self.cfg['training']['cfg_path'] = self.cfg_path

        if 'holdout' in self.cfg:
            self.cfg['holdout']['cfg_path'] = self.cfg_path
            scorers = self.cfg['holdout'].get('scorers')
            if scorers:
                valid_scorers = {
                    'balanced_accuracy': balanced_accuracy_score,
                    'accuracy': accuracy_score,
                    'roc_auc': roc_auc_score,
                    'recall': recall_score,
                    'specificity': specificity,
                }
                self.cfg['holdout']['scorers'] = [valid_scorers[s] for s in scorers]
