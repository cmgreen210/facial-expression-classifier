from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np


class ClassifierBase(object):
    """

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        pass

    @abstractmethod
    def predict(self, x):
        """

        :param x:
        :return:
        """
        pass

    @abstractmethod
    def predict_proba(self, x):
        """

        :param x:
        :return:
        """
        pass


class DummyClassifier(ClassifierBase):
    """

    """
    def __init__(self, n):
        self._n = n

    def fit(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        pass

    def predict(self, x):
        """

        :param x:
        :return:
        """
        if self._n is None:
            raise StandardError('You must call fit before predict!')
        return np.random.randint(0, self._n, size=x.shape[0])

    def predict_proba(self, x):
        """
        :param x:
        :return:
        """
        if self._n is None:
            raise StandardError('You must call fit before predict!')
        m = x.shape[0]
        return (1 / self._n) * np.ones((m, self._n))

ClassifierBase.register(DummyClassifier)
