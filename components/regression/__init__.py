__author__ = 'feurerm'

from collections import OrderedDict
import importlib
import inspect
import os
import pkgutil
import sys

from ..base import AutoSklearnRegressionAlgorithm
from ..base import find_components, ThirdPartyComponents
from ..base import add_component_deepcopy
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

regression_dir = os.path.split(__file__)[0]
_regressors = find_components(__package__,
                              regression_dir,
                              AutoSklearnRegressionAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnRegressionAlgorithm)

def add_regressor(regressor):
    _addons.add_component(regressor)


class RegressorChoice(object):
    def __init__(self, **params):
        choice = params['__choice__']
        del params['__choice__']
        self.choice = self.get_components()[choice](**params)

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_regressors)
        components.update(_addons.components)
        return components


    @classmethod
    def get_available_components(cls, data_prop,
                                 include=None,
                                 exclude=None):
        components_ = cls.get_components()
        available = set(components_)
        components_dict = {}

        if include is not None and exclude is not None:
            raise ValueError("The argument include and exclude "
                             "cannot be used together.")

        if include is not None:
            for incl in include:
                if incl not in components_:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)
            available = set(include)
        elif exclude is not None:
            available -= set(exclude)

        for name in available:
            entry = components_[name]
            # Avoid infinite loop
            if entry == RegressorChoice:
                continue
            if entry.get_properties()['handles_regression'] is False:
                continue
            components_dict[name] = entry

        components_dict = OrderedDict(sorted(components_dict.items()))
        return components_dict

    @classmethod
    def get_hyperparameter_search_space(cls, dataset_properties,
                                        default=None,
                                        include=None,
                                        exclude=None):
        if include is not None and exclude is not None:
            raise ValueError("The argument include and exclude cannot be used together.")

        cs = ConfigurationSpace()

        # Compile a list of all estimator objects for this problem
        available_estimators = cls.get_available_components(
            data_prop=dataset_properties,
            include=include,
            exclude=exclude)

        if len(available_estimators) == 0:
            raise ValueError("No regressors found")

        if default is None:
            defaults = ['random_forest', 'support_vector_regression'] + \
                list(available_estimators.keys())
            for default_ in defaults:
                if default_ in available_estimators:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        estimator = CategoricalHyperparameter('__choice__',
                                              list(available_estimators.keys()),
                                              default=default)
        cs.add_hyperparameter(estimator)
        for estimator_name in available_estimators.keys():
            estimator_configuration_space = available_estimators[
                estimator_name]. \
                get_hyperparameter_search_space(dataset_properties)
            cs = add_component_deepcopy(cs, estimator_name,
                                        estimator_configuration_space)

        return cs

