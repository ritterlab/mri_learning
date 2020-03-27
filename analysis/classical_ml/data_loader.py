import h5py
import functools
import numpy as np
import os
import pandas as pd
from data import Data

from copy import deepcopy
from nibabel import Nifti1Image
from nilearn.input_data import NiftiLabelsMasker
from nibabel import load as nb_load
from sklearn.model_selection import train_test_split


def map_labels(arr):
    values = np.unique(arr)
    mapping = dict(zip(values, np.arange(len(values))))
    series = pd.Series(arr).replace(mapping)
    return np.array(series)


def _read_table(table_path):
    """Open tabular files according to their extension"""
    table_path = os.path.abspath(table_path)
    extension = table_path.split('.')[-1]
    if extension == 'csv':
        return pd.read_csv(table_path)
    if extension == 'tsv':
        return pd.read_csv(table_path, sep='\t')
    if extension in ('xlsx', 'xls'):
        return pd.read_excel(table_path)
    else:
        raise TypeError


def _traverse_subj(bids_directory, subj_folder):
    """Traverse BIDS folder to get image locations"""
    subj_path = os.path.join(bids_directory, subj_folder)
    anat = os.path.join(subj_path, 'anat')
    images = os.listdir(anat)
    return os.path.join(anat, images[0]) if images else None


def _get_img_paths(bids_directory, label='label', tsv='participants.tsv', extra_column='gender'):
    """Given a BIDS folder, retrieve a df with the paths to each image""" #TODO extra column
    df = pd.read_csv(os.path.join(bids_directory, tsv), sep='\t')
    return pd.DataFrame({'path': [_traverse_subj(bids_directory, s) for s in df.subject],
                         'subject': df.subject,
                         'label': df[label],
                         extra_column: df[extra_column] if extra_column in df.columns else None
                         })


class DataLoader:
    def __init__(self,
                 cfg,
                 columns=None,
                 label='label',
                 stratify_cols=('label',),
                 test_size=None,
                 transform=None,
                 target_transform=map_labels,
                 n_samples=None,
                 use_atlas=True,
                 atlas_strategy='mean',
                 use_holdout=True,
                 mean_normalize=False,
                 minmax_normalize=True):
        self.cfg = cfg
        self.columns = columns
        self.label = label
        self.stratify_cols = list(stratify_cols)
        self.test_size = test_size
        self.transform = transform
        self.target_transform = target_transform
        self.n_samples = n_samples
        self.use_atlas = use_atlas
        self.atlas_strategy = atlas_strategy
        self.use_holdout = use_holdout
        self.mean_normalize = mean_normalize
        self.minmax_normalize = minmax_normalize

        self.stratify = None
        self.atlas = None

        # data extraction
        extract_method = self.get_extract_method()
        dic_data = extract_method()

        # feature transformation
        transforms = self.get_transforms()
        for func, args in transforms:
            dic_data = func(dic_data, **args)

        # data loading and normalization
        self.data = self.create_dataset(dic_data)
        if self.mean_normalize:
            self.data.mean_normalize()
        if self.minmax_normalize:
            self.data.minmax_normalize()

    def _train_test_apply(func):
        """Decorator to apply functions to train and test arrays"""
        @functools.wraps(func)
        def wrapper(self, arr_dic, arr='X'):
            arr_dic = deepcopy(arr_dic)
            train, test = arr, arr + '_test'
            arr_train = func(self, arr_dic[train])
            arr_dic[train] = arr_train
            if arr_dic.get(test) is not None:
                arr_test = func(self, arr_dic[test])
                arr_dic[test] = arr_test
            return arr_dic
        return wrapper

    def get_extract_method(self):
        read_mode = self.cfg['read_mode']
        if read_mode == 'BIDS':
            return self.extract_bids
        if read_mode == 'h5':
            return self.extract_h5
        if read_mode == 'table':
            return self.extract_table
        raise TypeError(f"Read mode {read_mode} not recognized")

    def get_transforms(self):
        transformations = []
        if self.cfg['read_mode'] == 'BIDS':
            transformations.append((self.load_niftii_paths, {}))
        if self.transform:
            transformations.append((self.apply_transform, {}))
        if self.use_atlas:
            transformations.append((self.apply_atlas, {}))

        transformations.append((self.flatten_images, {}))
        if self.target_transform:
            transformations.append((self.apply_target_transform, {'arr': 'y'}))

        return transformations

    def extract_table(self):
        """
        Creates a Dataset from a tabular data source
        :return: dict with 'X':features and 'y':labels
        """
        df = _read_table(self.cfg['paths']['table'])
        if self.columns:
            df = df[self.columns + [self.label]]
        if self.n_samples:
            df = df.sample(min(len(df), self.n_samples), random_state=0)
        X = np.array(df.drop(labels=self.label, axis=1))
        y = np.array(df[self.label])
        return {'X': X, 'y': y}

    def extract_bids(self):
        """
        Creates a Dataset from the images belonging to a specified BIDS folder
        :return: dict with 'X':features and 'y':labels
        """
        bids_path = self.cfg['paths']['data']
        df = _get_img_paths(bids_path, self.label).dropna()
        if self.n_samples:
            df = df.sample(min(len(df), self.n_samples), random_state=0)

        X = df.path
        y = np.array(df[self.label])
        self.stratify = df[self.stratify_cols]
        return {'X': X, 'y': y}

    def extract_h5(self):
        """
        Creates a Dataset from h5 files
        :return: dict with 'X':features and 'y':labels, as well as 'X_test' and 'y_test' if they exist
        """
        h5_paths = self.cfg['paths']['h5']
        train_h5 = h5py.File(h5_paths['train'], 'r')
        X_train = np.array(train_h5['X'])
        y_train = np.array(train_h5['y'])
        if h5_paths.get('test'):
            test_h5 = h5py.File(h5_paths['test'], 'r')
            X_test = np.array(test_h5['X'])
            y_test = np.array(test_h5['y'])
        else:
            X_test, y_test = None, None

        return {'X': X_train, 'y': y_train, 'X_test': X_test, 'y_test': y_test}

    @_train_test_apply
    def load_niftii_paths(self, paths):
        return np.array([nb_load(path) for path in paths])

    @_train_test_apply
    def flatten_images(self, images):
        return np.array([i.flatten() for i in images])

    @_train_test_apply
    def apply_atlas(self, images):
        """
        Brain parcellation from an image using an atlas and a feature reduction method
        :param images: iterable with list of files or with np.arrays
        :return: np.array of dimensions len(images) x number or atlas regions
        """
        atlas_filename = self.cfg['paths']['atlas']
        masker = NiftiLabelsMasker(labels_img=atlas_filename, strategy=self.atlas_strategy)
        if isinstance(images[0], np.ndarray):
            images = np.array([Nifti1Image(x, affine=np.eye(4)) for x in images])
        return masker.fit_transform(imgs=images)

    @_train_test_apply
    def apply_transform(self, images):
        """Applies any specified transforms and flattens the image"""
        return np.array([self.transform(img)for img in images])
    
    @_train_test_apply
    def apply_target_transform(self, y):
        return self.target_transform(y)

    def create_dataset(self, dic_data):
        if dic_data.get('X_test') is None and self.use_holdout:
            X_train, X_test, y_train, y_test = train_test_split(dic_data['X'], dic_data['y'],
                                                                test_size=self.test_size,
                                                                stratify=self.stratify,
                                                                shuffle=True,
                                                                random_state=0)
        else:
            X_train, y_train = dic_data['X'], dic_data['y']
            X_test, y_test = dic_data.get('X_test'), dic_data.get('y_test')

        return Data(X_train, y_train, X_test, y_test)


