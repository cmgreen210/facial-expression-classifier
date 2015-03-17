from fec.classifier.classifier_base import ClassifierBase
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class LogisticBaseModel(ClassifierBase):
    """Basic Logistic Classifier

    Parameters
    ----------
    :param pre_process_pipelin: sklearn pipeline
    """
    def __init__(self, pre_process_pipeline=None, **kwargs):
        if pre_process_pipeline is None:
            self._model = LogisticRegression(**kwargs)
        else:
            pre_process_pipeline.append(
                ('clf', LogisticRegression(**kwargs))
            )
            self._model = Pipeline(pre_process_pipeline)

    def fit(self, x, y):
        """Fit the logistic model

        :param x: feature matrix
        :param y: labels
        """
        self._model.fit(x, y)

    def predict(self, x):
        """Predict labels from inputted features

        :param x: array of features
        :return: array of class prediction
        """
        return self._model.predict(x)

    def predict_proba(self, x):
        """Predict classes and corresponding probabilities for features x

        :param x: array of features
        :return: classes and probabilities array
        """
        return self._model.predict(x)

    def save(self, path):
        """Save this model as pickle

        :param path: path to pickle model
        """
        joblib.dump(self._model, path)

    def load(self, path):
        """Load pickled model to this one

        :param path: path of already pickled model
        """
        self._model = joblib.load(path)


if __name__ == '__main__':
    data_path = "data/django_expression.pkl"
    df = pd.read_pickle(data_path)

    cond_happy = df['django_expression'] == 3
    cond_sad = df['django_expression'] == 4
    cond_surprise = df['django_expression'] == 5

    df = df[cond_happy | cond_sad | cond_surprise]

    x = np.array(df['pixels'].tolist())
    y = np.array(df['django_expression'].values)

    row_means = np.mean(x, axis=1)
    x -= row_means[:, np.newaxis]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8)

    start_pipeline = [('scaler', StandardScaler()),
                      ('pca', PCA(n_components=128))]
    log_clf = LogisticBaseModel(start_pipeline, **{'C': 0.5})
    log_clf.fit(xtrain, ytrain)

    train_accuracy = accuracy_score(ytrain, log_clf.predict(xtrain))
    test_accuracy = accuracy_score(ytest, log_clf.predict(xtest))

    print "Training accuracy: {0}".format(train_accuracy)
    print "Testing accuracy:  {0}".format(test_accuracy)

    log_clf.save('data/log_model.pkl')

    log_clf2 = LogisticBaseModel()
    log_clf2.load('data/log_model.pkl')

    train_accuracy = accuracy_score(ytrain, log_clf2.predict(xtrain))
    test_accuracy = accuracy_score(ytest, log_clf2.predict(xtest))

    print "Training accuracy: {0}".format(train_accuracy)
    print "Testing accuracy:  {0}".format(test_accuracy)
