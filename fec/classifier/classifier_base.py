from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np


class ClassifierBase(object):
    """Abstract base class for facial emotion classifiers

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, x, y):
        """Abstract fit method

        :param x: feature matrix
        :param y: array of class labels
        :return:
        """
        pass

    @abstractmethod
    def predict(self, x):
        """Predict classes from features

        :param x: feature matrix
        :return:
        """
        pass

    @abstractmethod
    def predict_proba(self, x):
        """Predict classes as well as return prediction probabilities

        :param x:
        :return:
        """
        pass


class DummyClassifier(ClassifierBase):
    """Dummy Clasiifier for testings
    """
    def __init__(self, n):
        self._n = n

    def fit(self, x, y):
        """Method does nothing!

        :param x:
        :param y:
        :return:
        """
        pass

    def predict(self, x):
        """Method does nothing!

        :param x:
        :return:
        """
        if self._n is None:
            raise StandardError('You must call fit before predict!')
        return np.random.randint(0, self._n, size=x.shape[0])

    def predict_proba(self, x):
        """Predicts uniform probability
        :param x:
        :return:
        """
        if self._n is None:
            raise StandardError('You must call fit before predict!')
        m = x.shape[0]
        return (1 / self._n) * np.ones((m, self._n))

ClassifierBase.register(DummyClassifier)
