from sklearn.dummy import DummyClassifier, DummyRegressor
import numpy as np


class MyDummyClassifier(DummyClassifier):
    def __init__(self, configuration, random_states):
        super(MyDummyClassifier, self).__init__(strategy="most_frequent")

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X, y, sample_weight=None):
        return super(MyDummyClassifier, self).fit(np.ones((X.shape[0], 1)), y,
                                                  sample_weight=sample_weight)

    def fit_estimator(self, X, y, fit_params=None):
        return self.fit(X, y)

    def predict_proba(self, X, batch_size=1000):
        new_X = np.ones((X.shape[0], 1))
        return super(MyDummyClassifier, self).predict_proba(new_X)

    def estimator_supports_iterative_fit(self):
        return False


class MyDummyRegressor(DummyRegressor):
    def __init__(self, configuration, random_states):
        super(MyDummyRegressor, self).__init__(strategy='mean')

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X, y, sample_weight=None):
        return super(MyDummyRegressor, self).fit(np.ones((X.shape[0], 1)), y,
                                                 sample_weight=sample_weight)

    def fit_estimator(self, X, y, fit_params=None):
        return self.fit(X, y)

    def predict(self, X, batch_size=1000):
        new_X = np.ones((X.shape[0], 1))
        return super(MyDummyRegressor, self).predict(new_X)

    def estimator_supports_iterative_fit(self):
        return False
