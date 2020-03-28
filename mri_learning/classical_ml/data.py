import numpy as np


class Data:
    """This class stores the data X_train and y_train as a training set,
    and if specified X_test and y_test are use as a holdout set"""

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.mean_normalized = False
        self.minmax_normalized = False

    def merge(self, other):
        """Joins this Data object with another one horizontally to expand the number of features.
        This method assumes that both objects have the same row order.
        Args: `Data` object to be merged with the current one
        Returns: `Data` object resulting from the merge of `self` and `other`
        """
        X_train = np.hstack([self.X_train, other.X_train])
        y_train = self.y_train
        if self.X_test is not None and self.y_test is not None:
            X_test = np.hstack([self.X_test, other.X_test])
            y_test = self.y_test
        else:
            X_test, y_test = None, None
        return Data(X_train, y_train, X_test, y_test)

    def mean_normalize(self):
        """Normalization to get mean 0 and std 1"""
        if self.mean_normalized:
            print('Data was already normalized')
        else:
            mean, std = self.X_train.mean(axis=0), self.X_train.std(axis=0)
            self.X_train = (self.X_train - mean) / (std + 1e-06)
            if self.X_test is not None:
                self.X_test = (self.X_test - mean) / (std + 1e-06)
            self.mean_normalized = True

    def minmax_normalize(self):
        """Normalize to get into a range from 0 to 1"""
        if self.minmax_normalized:
            print('Data was already normalized')
        else:
            for i in range(len(self.X_train)):
                self.X_train[i] -= np.min(self.X_train[i])
                self.X_train[i] /= np.max(self.X_train[i])

            if self.X_test is not None:
                for i in range(len(self.X_test)):
                    self.X_test[i] -= np.min(self.X_test[i])
                    self.X_test[i] /= np.max(self.X_test[i])

            self.minmax_normalized = True

    def __len__(self):
        return len(self.y_train) if self.y_test is None else len(self.y_train) + len(self.y_test)

    def __str__(self):
        return f"Data Object of {len(self)} samples"

    def __repr__(self):
        return self.__str__()