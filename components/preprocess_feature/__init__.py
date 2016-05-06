__author__ = 'feurerm'

from collections import OrderedDict
import importlib
import inspect
import os
import pkgutil
import sys

from ..base import AutoSklearnPreprocessingAlgorithm
from ..base import find_components, ThirdPartyComponents
from ..base import add_component_deepcopy
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


preprocess_feature_dir = os.path.split(__file__)[0]
_preprocessors = find_components(__package__,
                                 preprocess_feature_dir,
                                 AutoSklearnPreprocessingAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)


def add_preprocessor(preprocessor):
    _addons.add_component(preprocessor)


class FeaturePreprocessorChoice(object):
    def __init__(self, **params):
        choice = params['__choice__']
        del params['__choice__']
        self.choice = self.get_components()[choice](**params)

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_preprocessors)
        components.update(_addons.components)
        return components

    @classmethod
    def get_available_components(cls, data_prop,
                                 include=None,
                                 exclude=None):
        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = cls.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        # TODO check for task type classification and/or regression!

        components_dict = {}
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == FeaturePreprocessorChoice or hasattr(entry, 'get_components'):
                continue

            target_type = data_prop['target_type']
            if target_type == 'classification':
                if entry.get_properties()['handles_classification'] is False:
                    continue
                if data_prop.get('multiclass') is True and \
                        entry.get_properties()['handles_multiclass'] is False:
                    continue
                if data_prop.get('multilabel') is True and \
                        entry.get_properties()['handles_multilabel'] is False:
                    continue
            elif target_type == 'regression':
                if entry.get_properties()['handles_regression'] is False:
                    continue
            else:
                raise ValueError('Unknown target type %s' % target_type)
            components_dict[name] = entry

        components_dict = OrderedDict(sorted(components_dict.items()))
        return components_dict

    @classmethod
    def get_hyperparameter_search_space(cls, dataset_properties,
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
                "No preprocessors found, please add no_preprocessing")

        if default is None:
            defaults = ['no_preprocessing', 'select_percentile', 'pca',
                        'truncatedSVD']
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
