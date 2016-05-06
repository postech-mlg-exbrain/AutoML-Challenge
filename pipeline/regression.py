from collections import OrderedDict

import numpy as np
from sklearn.base import RegressorMixin

from pipeline.base import BasePipeline


class SimpleRegressionPipeline(RegressorMixin, BasePipeline):
    """This class implements the regression task.

    It implements a pipeline, which includes one preprocessing step and one
    regression algorithm. It can render a search space including all known
    regression and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available regressors at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    ConfigSpace.configuration_space.Configuration.

    Parameters
    ----------
    configuration : ConfigSpace.configuration_space.Configuration
        The configuration to evaluate.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    _estimator : The underlying scikit-learn regression model. This
        variable is assigned after a call to the
        :meth:`autosklearn.pipeline.regression.SimpleRegressionPipeline.fit`
        method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`autosklearn.pipeline.regression.SimpleRegressionPipeline.fit`
        method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """

    def __init__(self, configuration, task, random_state=None):
        super(SimpleRegressionPipeline, self).__init__(configuration,
                                                       task, random_state)

    def pre_transform(self, X, Y, fit_params=None, init_params=None):
        X, fit_params = super(SimpleRegressionPipeline, self).pre_transform(
            X, Y, fit_params=fit_params, init_params=init_params)
        self.num_targets = 1 if len(Y.shape) == 1 else Y.shape[1]
        return X, fit_params

    def fit_estimator(self, X, y, fit_params=None):
        self.y_max_ = np.nanmax(y)
        self.y_min_ = np.nanmin(y)
        return super(SimpleRegressionPipeline, self).fit_estimator(
            X, y, fit_params=fit_params)

    def iterative_fit(self, X, y, fit_params=None, n_iter=1):
        self.y_max_ = np.nanmax(y)
        self.y_min_ = np.nanmin(y)
        return super(SimpleRegressionPipeline, self).iterative_fit(
            X, y, fit_params=fit_params, n_iter=n_iter)

    def predict(self, X, batch_size=None):
        y = super(SimpleRegressionPipeline, self).\
            predict(X, batch_size=batch_size)
        y[y > (2 * self.y_max_)] = 2 * self.y_max_
        if self.y_min_ < 0:
            y[y < (2 * self.y_min_)] = 2 * self.y_min_
        elif self.y_min_ > 0:
            y[y < (0.5 * self.y_min_)] = 0.5 * self.y_min_
        return y

    @classmethod
    def get_available_components(cls, available_comp, data_prop, inc, exc):
        components_dict = {}
        for name in available_comp:
            if inc is not None and name not in inc:
                continue
            elif exc is not None and name in exc:
                continue
            entry = available_comp[name]

            if not entry.get_properties()['handles_regression']:
                continue
            components_dict[name] = entry

        components_dict = OrderedDict(sorted(components_dict.items()))
        return components_dict

    def _get_estimator_hyperparameter_name(self):
        return "regressor"