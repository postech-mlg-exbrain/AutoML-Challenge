import copy
from collections import OrderedDict
import importlib
import inspect
import pkgutil
import sys

from ConfigSpace.conditions import EqualsCondition


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(
                        obj) and base_class in obj.__bases__:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class ThirdPartyComponents(object):
    def __init__(self, base_class):
        self.base_class = base_class
        self.components = OrderedDict()

    def add_component(self, obj):
        if inspect.isclass(obj) and self.base_class in obj.__bases__:
            name = obj.__name__
            classifier = obj
        else:
            raise TypeError('add_component works only with a subclass of %s' %
                            str(self.base_class))

        properties = set(classifier.get_properties())
        should_be_there = {'shortname', 'name', 'handles_regression',
                           'handles_classification', 'handles_multiclass',
                           'handles_multilabel', 'is_deterministic',
                           'input', 'output'}
        for property in properties:
            if property not in should_be_there:
                raise ValueError('Property %s must not be specified for '
                                 'algorithm %s. Only the following properties '
                                 'can be specified: %s' %
                                 (property, name, str(should_be_there)))
        for property in should_be_there:
            if property not in properties:
                raise ValueError('Property %s not specified for algorithm %s')

        self.components[name] = classifier
        print(name, classifier)


class AutoSklearnClassificationAlgorithm(object):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/components/classification` to make it available."""

    def __init__(self):
        self.estimator = None
        self.properties = None

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of the underlying algorithm. These are:

        * Short name
        * Full name
        * Can the algorithm handle missing values?
          (handles_missing_values : {True, False})
        * Can the algorithm handle nominal features?
          (handles_nominal_features : {True, False})
        * Can the algorithm handle numerical features?
          (handles_numerical_features : {True, False})
        * Does the algorithm prefer data scaled in [0,1]?
          (prefers_data_scaled : {True, False}
        * Does the algorithm prefer data normalized to 0-mean, 1std?
          (prefers_data_normalized : {True, False}
        * Can the algorithm handle multiclass-classification problems?
          (handles_multiclass : {True, False})
        * Can the algorithm handle multilabel-classification problems?
          (handles_multilabel : {True, False}
        * Is the algorithm deterministic for a given seed?
          (is_deterministic : {True, False)
        * Can the algorithm handle sparse data?
          (handles_sparse : {True, False}
        * What are the preferred types of the data array?
          (preferred_dtype : list of tuples)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Return the configuration space of this classification algorithm.

        Returns
        -------
        ConfigSpace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def fit(self, X, y):
        """The fit function calls the fit function of the underlying
        scikit-learn model and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,) or shape = (n_sample, n_labels)

        Returns
        -------
        self : returns an instance of self.
            Targets

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def predict(self, X):
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,) or shape = (n_sample, n_labels)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def predict_proba(self, X):
        """Predict probabilities.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        raise NotImplementedError()

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator

    def __str__(self):
        name = self.get_properties()['name']
        return "autosklearn.pipeline %s" % name


class AutoSklearnPreprocessingAlgorithm(object):
    """Provide an abstract interface for preprocessing algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/components/preprocessing` to make it available."""

    def __init__(self):
        self.preprocessor = None

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of the underlying algorithm. These are:

        * Short name
        * Full name
        * Can the algorithm handle missing values?
          (handles_missing_values : {True, False})
        * Can the algorithm handle nominal features?
          (handles_nominal_features : {True, False})
        * Can the algorithm handle numerical features?
          (handles_numerical_features : {True, False})
        * Does the algorithm prefer data scaled in [0,1]?
          (prefers_data_scaled : {True, False}
        * Does the algorithm prefer data normalized to 0-mean, 1std?
          (prefers_data_normalized : {True, False}
        * Can preprocess regression data?
          (handles_regression : {True, False}
        * Can preprocess classification data?
          (handles_classification : {True, False}
        * Can the algorithm handle multiclass-classification problems?
          (handles_multiclass : {True, False})
        * Can the algorithm handle multilabel-classification problems?
          (handles_multilabel : {True, False}
        * Is the algorithm deterministic for a given seed?
          (is_deterministic : {True, False)
        * Can the algorithm handle sparse data?
          (handles_sparse : {True, False}
        * What are the preferred types of the data array?
          (preferred_dtype : list of tuples)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Return the configuration space of this preprocessing algorithm.

        Returns
        -------
        ConfigSpace.configuration_space.ConfigurationSpace
            The configuration space of this preprocessing algorithm.
        """
        raise NotImplementedError()

    def fit(self, X, Y):
        """The fit function calls the fit function of the underlying
        scikit-learn preprocessing algorithm and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,) or shape = (n_samples, n_labels)

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def transform(self, X):
        """The transform function calls the transform function of the
        underlying scikit-learn model and returns the transformed array.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        X : array
            Return the transformed training data

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def get_preprocessor(self):
        """Return the underlying preprocessor object.

        Returns
        -------
        preprocessor : the underlying preprocessor object
        """
        return self.preprocessor

    def __str__(self):
        name = self.get_properties()['name']
        return "autosklearn.pipeline %s" % name


class AutoSklearnRegressionAlgorithm(object):
    """Provide an abstract interface for regression algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/pipeline/components/regression` to make it available."""

    def __init__(self):
        self.estimator = None
        self.properties = None

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of the underlying algorithm. These are:

        * Short name
        * Full name
        * Can the algorithm handle missing values?
          (handles_missing_values : {True, False})
        * Can the algorithm handle nominal features?
          (handles_nominal_features : {True, False})
        * Can the algorithm handle numerical features?
          (handles_numerical_features : {True, False})
        * Does the algorithm prefer data scaled in [0,1]?
          (prefers_data_scaled : {True, False}
        * Does the algorithm prefer data normalized to 0-mean, 1std?
          (prefers_data_normalized : {True, False}
        * Is the algorithm deterministic for a given seed?
          (is_deterministic : {True, False)
        * Can the algorithm handle sparse data?
          (handles_sparse : {True, False}
        * What are the preferred types of the data array?
          (preferred_dtype : list of tuples)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Return the configuration space of this regression algorithm.

        Returns
        -------
        ConfigSpace.configuration_space.ConfigurationSpace
            The configuration space of this regression algorithm.
        """
        raise NotImplementedError()

    def fit(self, X, y):
        """The fit function calls the fit function of the underlying
        scikit-learn model and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = [n_samples]

        Returns
        -------
        self : returns an instance of self.
            Targets

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def predict(self, X):
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator

    def __str__(self):
        name = self.get_properties()['name']
        return "autosklearn.pipeline %s" % name



def add_component_deepcopy(config_space, prefix, component_space):
    # We have to retrieve the configuration space every time because
    # we change the objects it returns. If we reused it, we could not
    # retrieve the conditions further down
    # TODO implement copy for hyperparameters and forbidden and
    # conditions!
    component = config_space.get_hyperparameter("__choice__")
    for parameter in component_space.get_hyperparameters():
        new_parameter = copy.deepcopy(parameter)
        new_parameter.name = "%s:%s" % (prefix,
                                        new_parameter.name)
        config_space.add_hyperparameter(new_parameter)
        # We must only add a condition if the hyperparameter is not
        # conditional on something else
        if len(component_space.get_parents_of(parameter)) == 0:
            condition = EqualsCondition(new_parameter, component,
                                        prefix)
            config_space.add_condition(condition)
    space_copy = copy.deepcopy(component_space)
    for condition in space_copy.get_conditions():
        dlcs = condition.get_descendant_literal_conditions()
        for dlc in dlcs:
            if not dlc.child.name.startswith(prefix):
                dlc.child.name = "%s:%s" % (
                    prefix, dlc.child.name)
            if not dlc.parent.name.startswith(prefix):
                dlc.parent.name = "%s:%s" % (
                    prefix, dlc.parent.name)
        config_space.add_condition(condition)
    space_copy = copy.deepcopy(component_space)
    for forbidden_clause in space_copy.forbidden_clauses:
        dlcs = forbidden_clause.get_descendant_literal_clauses()
        for dlc in dlcs:
            if not dlc.hyperparameter.name.startswith(prefix):
                dlc.hyperparameter.name = "%s:%s" % (prefix,
                                                     dlc.hyperparameter.name)
        config_space.add_forbidden_clause(forbidden_clause)

    return config_space

