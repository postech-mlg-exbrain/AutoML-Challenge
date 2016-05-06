import ast
import logging

from .base import MetaBase
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from kND import KNearestDatasets

class MetaLearningOptimizer(object):
    def __init__(self, name, configuration_space, meta_dir, logger,
                 seed=None,
                 use_features="",
                 metric='l1',
                 metric_kwargs=None,
                 subset='all'):
        self.name = name
        self.configuration_space = configuration_space
        self.meta_dir = meta_dir
        self.seed = seed
        self.use_features = use_features
        self.metric = metric
        self.distance_kwargs = metric_kwargs
        self.subset = subset

        self.kND = None
        self.meta_base = MetaBase(configuration_space, meta_dir, logger)
        self.logger = logger

    def suggest_all(self, exclude_double_config=True):
        """Return a list of the best hyperparameters of neighboring datasets"""
        # TODO check if _learn was called before!
        neighbors = self._learn(exclude_double_config)
        hp_list = []
        for neighbor in neighbors:
            try:
                configuration = \
                    self.meta_base.get_configuration_from_algorithm_index(
                    neighbor[2])
                #self.logger.info("%s %s %s" % (neighbor[0], neighbor[1], configuration))
            except (KeyError):
                self.logger.warning("Configuration %s not found" % neighbor[2])
                continue

            hp_list.append(configuration)
        return hp_list

    def _get_metafeatures(self):
        """This is inside an extra function for testing purpose"""
        # Load the task

        all_metafeatures = self.meta_base.get_all_metafeatures()

        # TODO: buggy and hacky, replace with a list seperated by commas
        if self.use_features and \
                (type(self.use_features) != str or self.use_features != ''):
            if type(self.use_features) == str:
                use_features = self.use_features.split(",")
            elif type(self.use_features) in (list, np.ndarray):
                use_features = self.use_features
            else:
                raise NotImplementedError(type(self.use_features))

            if len(use_features) == 0:
                self.logger.info("You just tried to remove all metafeatures...")
            else:
                keep = [col for col in all_metafeatures.columns if col in use_features]
                all_metafeatures = all_metafeatures.loc[:,keep]
                self.logger.info("Going to keep the following metafeatures:")
                self.logger.info(str(keep))

        return self._split_metafeature_array(self.name, all_metafeatures)


    def _split_metafeature_array(self, dataset_name, metafeatures):
        """Split the metafeature array into dataset metafeatures and all other.

        This is inside an extra function for testing purpose.
        """
        dataset_metafeatures = metafeatures.loc[dataset_name].copy()
        metafeatures = metafeatures[metafeatures.index != dataset_name]
        return dataset_metafeatures, metafeatures



    def _learn(self, exclude_double_configurations=True):
        dataset_metafeatures, all_other_metafeatures = self._get_metafeatures()

        # Remove metafeatures which could not be calculated for the target
        # dataset
        keep = []
        for idx in dataset_metafeatures.index:
            if np.isfinite(dataset_metafeatures.loc[idx]):
               keep.append(idx)

        dataset_metafeatures = dataset_metafeatures.loc[keep]
        all_other_metafeatures = all_other_metafeatures.loc[:,keep]

        # Do mean imputation of all other metafeatures
        all_other_metafeatures = all_other_metafeatures.fillna(
            all_other_metafeatures.mean())

        if self.kND is None:
            # In case that we learn our distance function, get_value the parameters for
            #  the random forest
            if self.distance_kwargs:
                rf_params = ast.literal_eval(self.distance_kwargs)
            else:
                rf_params = None

            # To keep the distance the same in every iteration, we create a new
            # random state
            random_state = check_random_state(self.seed)
            kND = KNearestDatasets(metric=self.metric,
                                   random_state=random_state,
                                   metric_params=rf_params)

            runs = dict()
            # TODO move this code to the metabase
            for task_id in all_other_metafeatures.index:
                try:
                    runs[task_id] = self.meta_base.get_runs(task_id)
                except KeyError:
                    # TODO should I really except this?
                    self.logger.warning("Could not find runs for instance %s" % task_id)
                    runs[task_id] = pd.Series([], name=task_id)
            runs = pd.DataFrame(runs)

            kND.fit(all_other_metafeatures, runs)
            self.kND = kND
        return self.kND.kBestSuggestions(dataset_metafeatures, k=-1,
            exclude_double_configurations=exclude_double_configurations)
