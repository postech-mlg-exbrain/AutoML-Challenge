import hashlib

from numpy.ma.core import _minimum_operation
from scipy.stats._continuous_distns import mielke_gen

from constants import *
from six import string_types

import numpy as np
from scipy.sparse import issparse
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
#from components.implementations.OneHotEncoder import OneHotEncoder

from metalearning.utils import calculate_metafeatures
from metalearning.utils import EXCLUDE_META_REGRESSION, EXCLUDE_META_CLASSIFICATION

_EXPANSION_THRESHOLD = 128 * 1024 * 1024

def estimate_1hot_cost(X, is_categorical):
    """
    Calculate the "memory expansion" after applying one-hot encoding.

    :param X: array-like
        The input data array
    :param is_categorical: boolean array-like
        Array of vector form that indicates
        whether each features of X is categorical

    :return: int
        Calculated memory size in byte scale (expansion)
    """
    n_columns = 0
    count_labels_v = lambda v: np.sum(np.isfinite(np.unique(v))) - 1
    n_labels = np.apply_along_axis(count_labels_v, 0, X)
    n_columns += np.sum(n_labels[is_categorical])

    estimated_memory = n_columns * X.shape[0] * X.dtype.itemsize
    return estimated_memory


class DataManager():

    def __init__(self, name):
        self._data = dict()
        self._info = dict()
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def info(self):
        # type: () -> object
        return self._info

    @property
    def feat_type(self):
        return self._feat_type

    @feat_type.setter
    def feat_type(self, value):
        self._feat_type = value

    @property
    def is_categorical(self):
        return self._is_categorical

    @is_categorical.setter
    def is_categorical(self, value):
        self._is_categorical = value

    @property
    def encoder(self):
        return self._encoder

    def encode_X(self, X, trans_only=False):
        # Currently : Imputer, OneHotEncoder
        if X is None:
            raise ValueError('encode_X can only be called when '
                             'data is given')
        if not hasattr(self, 'feat_type'):
            raise ValueError('encode_1hot can only be called on '
                             'data which has feat_type')

        sparse = self.info['is_sparse']

        if trans_only:
            if self.encoder is None:
                raise ValueError("encoder must be initialized "
                                 "before the transformation")
            for enc in self.encoder:
                X = enc.transform(X)
            if not sparse and issparse(X):
                X = X.todense()
            return X

        if hasattr(self, 'encoder'):
            raise ValueError('Non-trans-only encode_1hot can only be called on '
                             'non-encoded data. Use trans_only=True')

        self._encoder = None
        steps = []

        # Complete the data when it has some missing values
        # TODO Fine algorithm..! (Numerical vs Categorical)
        if self.info["has_missing"]:
            mc = Imputer(strategy="median", copy=False)
            steps.append(mc)

        vth = VarianceThreshold()
        steps.append(vth)

        # 1-hot encoding
        onehot_target = ["Categorical"]
        onehot_mask = np.array([f in onehot_target for f in self.feat_type])

        if not sparse:
            mem_expansion = float(estimate_1hot_cost(X, self.is_categorical))
            if mem_expansion > _EXPANSION_THRESHOLD:
                sparse = True

        # Assumption that every categorical columns ranges in [0, n_class)
        if any(onehot_mask):
            ohe = OneHotEncoder(categorical_features=onehot_mask,
                                dtype=np.float32, sparse=sparse)
            steps.append(ohe)

        for enc in steps:
            X = enc.fit_transform(X)

        if not sparse and issparse(X):
            X = X.todense()

        self._encoder = steps
        self.info['is_sparse'] = sparse

        return X


    def metafeatures(self, X, y):
        task = self.info['task']
        if task not in TASK_TYPES:
            return None

        X_ = X.copy()

        is_categorical = self.is_categorical
        encoded = hasattr(self, 'encoder')
        if encoded:
            is_categorical = [False]*(X_.shape[1])

        dont_calculate = set([])
        if task in CLASSIFICATION_TASKS:
            dont_calculate |= EXCLUDE_META_CLASSIFICATION
        elif task in REGRESSION_TASKS:
            dont_calculate |= EXCLUDE_META_REGRESSION
        else:
            raise NotImplementedError()

        result = calculate_metafeatures(X_, y,
                                        is_categorical=is_categorical,
                                        name=self.name,
                                        encoded=encoded,
                                        dont_calculate=dont_calculate)
        return result

    def __repr__(self):
        return 'DataManager : ' + self.name

    def __str__(self):
        val = 'DataManager : ' + self.name + '\ninfo:\n'
        for item in self.info:
            val += '\t' + item + ' = ' + str(self.info[item]) + '\n'
        val += 'data:\n'

        for subset in self.data:
            val += '\t%s = %s %s %s\n' % (subset, type(self.data[subset]),
                                          str(self.data[subset].shape),
                                          str(self.data[subset].dtype))
            if issparse(self.data[subset]):
                val += '\tdensity: %f\n' % \
                       (float(len(self.data[subset].data))
                        / self.data[subset].shape[0]
                        / self.data[subset].shape[1])
        val += 'feat_type:\t' + str(self.feat_type) + '\n'
        return val


def get_datamanager(X, y, info, feat_type):
    # TODO unsupervised learning i.e. y==None

    task = info.get('task')
    metric = info.get('metric')
    name = info.get('name')
    is_sparse = info.get("is_sparse")
    has_missing = info.get('has_missing')
    has_categorical = info.get('has_categorical')
    feat_num = info.get('feat_num')

    # TODO assumption X : ndarray
    # Check and fill in the name
    if name is None:
        m = hashlib.md5()
        if issparse(X):
            m.update(str(X[:256, :32]))
        else:
            m.update(X[:256, :32].tostring())
        name = m.hexdigest()
        info['name'] = name

    if task is None:
        # TODO Logic for this inference
        task = 'multiclass.classification'
        info['task'] = task

    if metric is None:
        if task in STRING_CLASSIFICATION:
            metric = 'acc_metric'
            info['metric'] = metric
        elif task in STRING_REGRESSION:
            metric = 'r2_metric'
            info['metric'] = metric
        else:
            raise NotImplementedError()

    # TODO feat_type recognition: categorical vs ordinal
    if feat_type is None:
        feat_type = ['Numerical'] * X.shape[1]

    if is_sparse is None:
        is_sparse = issparse(X)
        info['is_sparse'] = is_sparse
    else:
        is_sparse = bool(is_sparse)

    if has_missing is None:
        if is_sparse:
            has_missing = ~np.all(np.isfinite(X.data))
        else:
            has_missing = ~np.all(np.isfinite(X))
        info['has_missing'] = has_missing
    else:
        has_missing = bool(has_missing)

    if has_categorical is None:
        has_categorical = False
        info['has_categorical'] = has_categorical
    else:
        has_categorical = bool(has_categorical)

    if feat_num is None:
        feat_num = X.shape[1]
        info['feat_num'] = feat_num

    train_num = X.shape[0]

    # Input validation (raises ValueError)
    if task not in STRING_TO_TASK_TYPES:
        raise ValueError('String attribute task must be one of '
                         '%s, while your input is %s'
                         % (str(STRING_TO_TASK_TYPES.keys()), task))
    task = STRING_TO_TASK_TYPES[task]

    if metric not in STRING_TO_METRIC:
        raise ValueError('String attribute metric must be one of '
                         '%s, while your input is %s'
                         % (str(STRING_TO_METRIC.keys()), metric))
    metric = STRING_TO_METRIC[metric]

    if task not in TASK_TYPES:
        raise ValueError('Invalid task: %s', str(task))

    if metric not in METRIC:
        raise ValueError('Invalid metric: %s', str(metric))

    if metric in REGRESSION_METRICS and \
                    info['task'] in CLASSIFICATION_TASKS:
        metric = ACC_METRIC

    if metric in CLASSIFICATION_METRICS and \
                    info['task'] in REGRESSION_METRICS:
        metric = R2_METRIC

    if feat_type is not None and not all([f in ["Numerical", "Categorical", "Binary"]
                                          for f in feat_type]):
        raise ValueError('Array feat_type must either Numerical or Categorical.')

    datamanager = DataManager(name)

    datamanager.info['task'] = task
    datamanager.info['metric'] = metric
    datamanager.info['is_sparse'] = is_sparse
    datamanager.info['has_missing'] = has_missing
    datamanager.info['feat_num'] = feat_num
    datamanager.info['train_num'] = train_num

    if task in CLASSIFICATION_TASKS:
        label_num = {
            BINARY_CLASSIFICATION: 2,
            MULTICLASS_CLASSIFICATION: len(unique_labels(y)),
            MULTILABEL_CLASSIFICATION: y.shape[-1]
        }
    elif task in REGRESSION_TASKS:
        label_num = {
            REGRESSION: 1
        }
    else:
        raise NotImplementedError()

    datamanager.info['label_num'] = label_num[task]

    datamanager.feat_type = feat_type
    datamanager.is_categorical = np.array([f == "Categorical"
                                           for f in feat_type])

    return datamanager