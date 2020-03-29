import argparse
import warnings

from configuration import Configuration
from data_loader import DataLoader
from numpy.random import seed as np_seed
from tester import Tester
from trainer import Trainer
from random import seed


class ClassicalML:
    def __init__(self, cfg_path):
        np_seed(0)
        seed(0)
        self.cfg_path = cfg_path
        cfg = Configuration(cfg_path).cfg
        # load data
        dataset = DataLoader(cfg, **cfg['dataset'])
        # fit models to data
        t = Trainer(**cfg['training'])
        self.val_results = t.run_models(dataset.data)
        # test data
        if 'holdout' in cfg:
            tester = Tester(**cfg['holdout'])
            tester.run(dataset.data.X_test, dataset.data.y_test)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="path to the configuration file")
    path = parser.parse_args().cfg_path
    cml = ClassicalML(path)
    results = cml.val_results

