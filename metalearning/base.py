from abc import ABCMeta, abstractmethod
from time import time
import logging

from scipy.sparse import issparse
import pandas as pd
from six import StringIO, string_types
import arff

from .aslib_simple import AlgorithmSelectionProblem
from ConfigSpace.configuration_space import Configuration

class MetaFeatureValue(object):
    def __init__(self, name, type_, fold, repeat, value, time, comment=""):
        self.name = name
        self.type_ = type_
        self.fold = fold
        self.repeat = repeat
        self.value = value
        self.time = time
        self.comment = comment

    def to_arff_row(self):
        if self.type_ == "METAFEATURE":
            value = self.value
        else:
            value = "?"

        return [self.name, self.type_, self.fold,
                self.repeat, value, self.time, self.comment]

    def __repr__(self):
        repr = "%s (type: %s, fold: %d, repeat: %d, value: %s, time: %3.3f, " \
               "comment: %s)"
        repr = repr % tuple(self.to_arff_row()[:4] +
                            [str(self.to_arff_row()[4])] +
                            self.to_arff_row()[5:])
        return repr


class AbstractMetaFeature(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.type_ = None

    @abstractmethod
    def _calculate(cls, X, y, categorical, metafeatures, helpers):
        pass

    def __call__(self, X, y, categorical=None, metafeatures=None, helpers=None):
        if categorical is None:
            categorical = [False for i in range(X.shape[1])]

        start_time = time()
        try:
            if issparse(X) and hasattr(self, "_calculate_sparse"):
                value = self._calculate_sparse(X, y, categorical, metafeatures, helpers)
            else:
                value = self._calculate(X, y, categorical, metafeatures, helpers)
            comment = ""
        except MemoryError as e:
            value = None
            comment = "Memory Error"
        end_time = time()

        return MetaFeatureValue(self.__class__.__name__, self.type_,
                                0, 0, value, end_time-start_time, comment=comment)


class MetaBase(object):
    def __init__(self, configuration_space, meta_dir, logger):
        self.configuration_space = configuration_space
        self.meta_dir = meta_dir
        self.logger = logger

        meta_reader = AlgorithmSelectionProblem(self.meta_dir, logger)
        self.metafeatures = meta_reader.metafeatures
        self.algorithm_runs = meta_reader.algorithm_runs
        self.configurations = meta_reader.configurations

        _configs = dict()
        for alg_id, config in self.configurations.items():
            try:
                _configs[alg_id] = \
                    (Configuration(configuration_space, values=config))
            except (ValueError, KeyError) as e:
                self.logger.error("Error reading configurations: %s", e)

        self.configurations = _configs

    def add_dataset(self, name, metafeatures):
        metafeatures.name = name
        if isinstance(metafeatures, DatasetMetafeatures):
            metafeatures = pd.Series(name=name,
                                     data={mf.name: mf.value for mf in
                                           metafeatures.values.values()})
        self.metafeatures = self.metafeatures.append(metafeatures)

        runs = pd.Series([], name=name)
        for metric in self.algorithm_runs.keys():
            self.algorithm_runs[metric].append(runs)

    def get_runs(self, dataset_name, performance_measure=None):
        """Return a list of all runs for a dataset."""
        if performance_measure is None:
            performance_measure = list(self.algorithm_runs.keys())[0]
        return self.algorithm_runs[performance_measure].loc[dataset_name]

    def get_all_runs(self, performance_measure=None):
        """Return a dictionary with a list of all runs"""
        if performance_measure is None:
            performance_measure = list(self.algorithm_runs.keys())[0]
        return self.algorithm_runs[performance_measure]

    def get_metafeatures(self, dataset_name):
        dataset_metafeatures = self.metafeatures.loc[dataset_name]
        return dataset_metafeatures

    def get_all_metafeatures(self):
        """Create a pandas DataFrame for the metadata of all datasets."""
        return self.metafeatures

    def get_configuration_from_algorithm_index(self, idx):
        return self.configurations[idx]
        #configuration = self.configurations[idx]
        #configuration = Configuration(self.configuration_space,
        # **configuration)
        #return configuration

    def get_algorithm_index_from_configuration(self, configuration):
        for idx in self.configurations.keys():
            if configuration == self.configurations[idx]:
                return idx

        raise ValueError(configuration)


class DatasetMetafeatures(object):
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def _get_arff(self):
        output = dict()
        output['relation'] = "metafeatures_%s" % (self.name)
        output['description'] = ""
        output['attributes'] = [('name', 'STRING'),
                                ('type', 'STRING'),
                                ('fold', 'NUMERIC'),
                                ('repeat', 'NUMERIC'),
                                ('value', 'NUMERIC'),
                                ('time', 'NUMERIC'),
                                ('comment', 'STRING')]
        output['data'] = []

        for key in sorted(self.values):
            output['data'].append(self.values[key].to_arff_row())
        return output

    def dumps(self):
        return self._get_arff()

    def dump(self, path_or_filehandle):
        output = self._get_arff()

        if isinstance(path_or_filehandle, string_types):
            with open(path_or_filehandle, "w") as fh:
                arff.dump(output, fh)
        else:
            arff.dump(output, path_or_filehandle)

    @classmethod
    def load(cls, path_or_filehandle):

        if isinstance(path_or_filehandle, string_types):
            with open(path_or_filehandle) as fh:
                input = arff.load(fh)
        else:
            input = arff.load(path_or_filehandle)

        dataset_name = input['relation'].replace('metafeatures_', '')
        metafeature_values = []
        for item in input['data']:
            mf = MetaFeatureValue(*item)
            metafeature_values.append(mf)

        return cls(dataset_name, metafeature_values)

    def __repr__(self, verbosity=0):
        repr = StringIO()
        repr.write("Metafeatures for dataset %s\n" % self.name)
        for name in self.values:
            if verbosity == 0 and self.values[name].type_ != "METAFEATURE":
                continue
            if verbosity == 0:
                repr.write("  %s: %s\n" %
                           (str(name), str(self.values[name].value)))
            elif verbosity >= 1:
                repr.write("  %s: %10s  (%10fs)\n" %
                           (str(name), str(self.values[
                                               name].value)[:10],
                            self.values[name].time))

            # Add the reason for a crash if one happened!
            if verbosity > 1 and self.values[name].comment:
                repr.write("    %s\n" % self.values[name].comment)

        return repr.getvalue()

    def keys(self):
        return self.values.keys()

    def __getitem__(self, item):
        return self.values[item]
