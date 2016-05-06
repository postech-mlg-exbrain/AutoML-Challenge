import numpy as np
import scipy.sparse as sp

from sklearn.base import ClassifierMixin

from pipeline.base import BasePipeline
from components.preprocess_data.balancing import Balancing


class SimpleClassificationPipeline(ClassifierMixin, BasePipeline):
    """This class implements the classification task.

    It implements a pipeline, which includes one preprocessing step and one
    classification algorithm. It can render a search space including all known
    classification and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
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
    _estimator : The underlying scikit-learn classification model. This
        variable is assigned after a call to the
        :meth:`autosklearn.pipeline.classification.SimpleClassificationPipeline
        .fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`autosklearn.pipeline.classification.SimpleClassificationPipeline
        .fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """

    def __init__(self, configuration, task, random_state=None):
        super(SimpleClassificationPipeline, self).__init__(configuration,
                                                           task, random_state)
        self._output_dtype = np.int32

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        self.num_targets = 1 if len(y.shape) == 1 else y.shape[1]

        # Weighting samples has to be done here, not in the components
        if self.configuration['balancing:strategy'] == 'weighting':
            balancing = Balancing(strategy='weighting')
            init_params, fit_params = balancing.get_weights(
                y, self.configuration['classifier:__choice__'],
                self.configuration['preprocessor:__choice__'],
                init_params, fit_params)

        X, fit_params = super(SimpleClassificationPipeline, self).pre_transform(
            X, y, fit_params=fit_params, init_params=init_params)

        return X, fit_params

    def predict_proba(self, X, batch_size=None):
        """predict_proba.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the pipeline will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """

        assert hasattr(self, 'pipeline_'), "fit() must be called " \
                                           "before call predict()"
        if batch_size is None:
            Xt = X
            for name, transform in self.pipeline_.steps[:-1]:
                Xt = transform.transform(Xt)
            return self.pipeline_.steps[-1][-1].predict_proba(Xt)
        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")
            else:
                # Probe for the target array dimensions
                target = self.predict_proba(X[0:2].copy())

                y = np.zeros((X.shape[0], target.shape[1]), dtype=np.float32)

                for k in range(max(1, int(np.ceil(float(X.shape[0])/batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = np.array(self.predict_proba(X[batch_from:batch_to],
                                                                         batch_size=None), dtype=np.float32)

                return y

    def _get_estimator_hyperparameter_name(self):
        return "classifier"

