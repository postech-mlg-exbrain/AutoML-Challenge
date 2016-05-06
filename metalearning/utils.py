import copy
from collections import deque
import os.path
import csv

from scipy.sparse import issparse
from sklearn.preprocessing import OneHotEncoder
from components.implementations.StandardScaler import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_array
import numpy as np

from features import metafeatures, helpers
from base import DatasetMetafeatures

SENTINEL = 'uiaeo'

EXCLUDE_META_FUTURES = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
}

META_MISSING_VALUES = {
    'NumberOfMissingValues',
    'NumberOfInstancesWithMissingValues',
    'NumberOfFeaturesWithMissingValues',
    'PercentageOfMissingValues',
    'PercentageOfInstancesWithMissingValues',
    'PercentageOfFeaturesWithMissingValues',
}

EXCLUDE_META_CLASSIFICATION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'PCA'
}

EXCLUDE_META_REGRESSION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'NumberOfClasses',
    'ClassOccurences',
    'ClassProbabilityMin',
    'ClassProbabilityMax',
    'ClassProbabilityMean',
    'ClassProbabilitySTD',
    'ClassEntropy',
    'LandmarkRandomNodeLearner',
    'PCA',
}

# TODO Regression Landmark ..!!

npy_metafeatures = {"LandmarkLDA", "LandmarkNaiveBayes", "LandmarkDecisionTree", "LandmarkDecisionNodeLearner",
                    "LandmarkRandomNodeLearner", "LandmarkWorstNodeLearner", "Landmark1NN",
                    "PCAFractionOfComponentsFor95PercentVariance", "PCAKurtosisFirstPC", "PCASkewnessFirstPC",
                    "Skewnesses", "SkewnessMin", "SkewnessMax", "SkewnessMean", "SkewnessSTD",
                    "Kurtosisses", "KurtosisMin", "KurtosisMax", "KurtosisMean", "KurtosisSTD"}

subsets = dict()
# All implemented metafeatures
subsets["all"] = set(metafeatures.functions.keys())

# Metafeatures used by Pfahringer et al. (2000) in the first experiment
subsets["pfahringer_2000_experiment1"] = {"number_of_features", "number_of_numeric_features",
                                          "number_of_categorical_features", "number_of_classes",
                                          "class_probability_max", "landmark_lda", "landmark_naive_bayes",
                                          "landmark_decision_tree"}

# Metafeatures used by Pfahringer et al. (2000) in the second experiment
# worst node learner not implemented yet
"""
pfahringer_2000_experiment2 = set(["landmark_decision_node_learner",
                                   "landmark_random_node_learner",
                                   "landmark_worst_node_learner",
                                   "landmark_1NN"])
"""

# Metafeatures used by Yogatama and Mann (2014)
subsets["yogotama_2014"] = {"log_number_of_features", "log_number_of_instances", "number_of_classes"}

# Metafeatures used by Bardenet et al. (2013) for the AdaBoost.MH experiment
subsets["bardenet_2013_boost"] = {"number_of_classes", "log_number_of_features", "log_inverse_dataset_ratio",
                                  "pca_95percent"}

# Metafeatures used by Bardenet et al. (2013) for the Neural Net experiment
subsets["bardenet_2013_nn"] = {"number_of_classes", "log_number_of_features", "log_inverse_dataset_ratio",
                               "pca_kurtosis_first_pc", "pca_skewness_first_pc"}

DENSIFY_THRESHOLD = 128


def calculate_metafeatures(X, y, is_categorical, name, encoded,
                           calculate=None, dont_calculate=None):
    name += SENTINEL

    if encoded:
        if any(is_categorical):
            raise ValueError("Training data may not 1-hot-encoded")
        calculate = set()
        calculate.update(npy_metafeatures)
        calculate -= dont_calculate
    else:
        if dont_calculate is None:
            dont_calculate = set()
        else:
            dont_calculate = copy.deepcopy(dont_calculate)
        dont_calculate.update(npy_metafeatures)

    metafeatures_ = copy.deepcopy(metafeatures)
    helpers_ = copy.deepcopy(helpers)

    metafeatures_.clear()
    helpers_.clear()
    mf_ = dict()

    visited = set()
    to_visit = deque()
    to_visit.extend(metafeatures_)

    X_trans = None

    # TODO calculate the numpy metafeatures after all others to consume less
    # TODO Parallelize this
    # memory
    while len(to_visit) > 0:
        name = to_visit.pop()
        if calculate is not None and name not in calculate:
            continue
        if dont_calculate is not None and name in dont_calculate:
            continue

        # encoded == True
        if name in npy_metafeatures:
            if X_trans is None:
                # TODO make sure this is done as efficient as possible (no copy for
                # sparse matrices because of wrong sparse format)
                sparse = issparse(X)
                sts = StandardScaler(copy=False, with_mean=(not sparse))
                pipeline = make_pipeline(sts)
                X_trans = pipeline.fit_transform(X)

                if not sparse and issparse(X_trans):
                    size_per_item_b =  X_trans.dtype.itemsize
                    n_elements = X_trans.shape[0] * X_trans.shape[1]
                    size_mb = n_elements * size_per_item_b / 1024 / 1024
                    if size_mb < DENSIFY_THRESHOLD:
                        X_trans = X_trans.todense()

                X_trans = check_array(X_trans,
                                      force_all_finite=True,
                                      accept_sparse='csr')
            X_ = X_trans
            y_ = y

        else:
            X_, y_ = X, y

        dependency = metafeatures.get_dependency(name)
        if dependency is not None:
            is_metafeature = dependency in metafeatures
            is_helper = dependency in helpers

            if is_metafeature and is_helper:
                raise NotImplementedError()
            elif not is_metafeature and not is_helper:
                raise ValueError(dependency)
            elif is_metafeature and not metafeatures_.is_calculated(dependency):
                to_visit.appendleft(name)
                continue
            elif is_helper and not helpers_.is_calculated(dependency):
                value = helpers_[dependency](X_, y_, is_categorical, metafeatures_, helpers_)
                helpers_.set_value(dependency, value)
                mf_[dependency] = value

        value = metafeatures_[name](X_, y_, is_categorical, metafeatures_, helpers_)
        metafeatures_.set_value(name, value)
        mf_[name] = value
        visited.add(name)

    mf_ = DatasetMetafeatures(name, mf_)
    return mf_







