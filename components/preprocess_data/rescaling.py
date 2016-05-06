from collections import OrderedDict

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from ..base import add_component_deepcopy
from constants import *


class Rescaling(object):
    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class NoRescalingComponent(Rescaling):
    def __init__(self, random_state):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MinMaxScaler',
                'name': 'MinMaxScaler',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}


class MinMaxScalerComponent(Rescaling):
    def __init__(self, random_state):
        from components.implementations.MinMaxScaler import MinMaxScaler
        self.preprocessor = MinMaxScaler()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MinMaxScaler',
                'name': 'MinMaxScaler',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}


class StandardScalerComponent(Rescaling):
    def __init__(self, random_state):
        from components.implementations.StandardScaler import StandardScaler
        self.preprocessor = StandardScaler()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'StandardScaler',
                'name': 'StandardScaler',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}


class NormalizerComponent(Rescaling):
    def __init__(self, random_state):
        from components.implementations.Normalizer import Normalizer
        self.preprocessor = Normalizer()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Normalizer',
                'name': 'Normalizer',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}


class RescalingChoice(object):
    def __init__(self, **params):
        choice = params['__choice__']
        del params['__choice__']
        self.choice = self.get_components()[choice](**params)

    @classmethod
    def get_components(cls):
        return OrderedDict((('none', NoRescalingComponent),
                            ('min/max', MinMaxScalerComponent),
                            ('standardize', StandardScalerComponent),
                            ('normalize', NormalizerComponent)))

    @classmethod
    def get_available_components(cls, data_prop=None,
                                 include=None,
                                 exclude=None):
        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = cls.get_components()

        components_dict = {}
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue
            entry = available_comp[name]
            components_dict[name] = entry

        components_dict = OrderedDict(sorted(components_dict.items()))
        return components_dict

    @classmethod
    def get_hyperparameter_search_space(cls, dataset_properties=None,
                                        default=None,
                                        include=None,
                                        exclude=None):
        cs = ConfigurationSpace()

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = cls.get_available_components(
            data_prop=dataset_properties,
            include=include, exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError(
                "No rescaling algorithm found.")

        if default is None:
            defaults = ['min/max', 'standardize', 'none', 'normalize']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter('__choice__',
                                                 list(
                                                     available_preprocessors.keys()),
                                                 default=default)
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[name]. \
                get_hyperparameter_search_space(dataset_properties)
            cs = add_component_deepcopy(cs, name,
                                        preprocessor_configuration_space)

        return cs

