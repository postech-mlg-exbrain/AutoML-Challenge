import logging
import os.path
import time
from collections import defaultdict
import gc
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, shuffle
import joblib

from constants import *
from datamanager import get_datamanager
from evaluation import _calculate_score, EvalProcess
from hyperspace import get_hyperspace
from metalearning.learner import MetaLearningOptimizer
from metalearning.utils import SENTINEL
from metalearning.utils import EXCLUDE_META_CLASSIFICATION, EXCLUDE_META_REGRESSION
from metalearning.utils import META_MISSING_VALUES
from metalearning.utils import subsets
from pipeline.utils import get_default_configs
from utils.stopwatch import StopWatch
from utils.process import memory_usage, handle_tasks
from utils.dumpmanager import DumpManager
from alice_exception import *


class AutoML(BaseEstimator):
    # TODO continuous multilabel
    # FIXME sparse config error
    # FIXME Submodule ConfigSpace
    # FIXME MAC memory_info
    # TODO pickle
    # TODO packaging
    # TODO repetitive update

    def __init__(self,
                 time_limit_total=1000,
                 time_limit_each=40,
                 memory_limit=None,
                 metalearning_k=10,
                 ensemble_size=1,
                 n_jobs=3,
                 random_state=1,
                 include_preprocessors=None,
                 resampling='cv',
                 log_filename=None,
                 tmp_dir=None,
                 debug=False):

        self.time_limit_total = time_limit_total
        self.time_limit_each = time_limit_each
        self.metalearning_k = metalearning_k
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.memory_limit = memory_limit
        self.random_state = random_state
        self.include_estimators = ("xgradient_boosting", "extra_trees")
        self.include_preprocessors = include_preprocessors
        self.mode = 'alice'
        self.resampling = resampling
        self.log_filename = log_filename
        self.tmp_dir = tmp_dir
        self.debug = debug

    def fit(self, X, y, valid_X=None, valid_y=None, info=None, feat_type=None):

        # Input validation (raises ValueError)
        if valid_X is not None and valid_y is not None:
            self._has_valid_data = True
        elif valid_X is not None or valid_y is not None:
            raise ValueError("validation data is given improperly")
        else:
            self._has_valid_data = False

        if self.mode not in MODES:
            raise ValueError("Invalid mode setting")
        if self.resampling not in RESAMPLING:
            raise ValueError("Invalid resampling strategy setting")
        if self._has_valid_data and self.resampling == 'cv':
            warnings.warn("cv does not use the predefined validation set. "
                          "Valid set will be ignored", RuntimeWarning)

        X, y = check_X_y(X, y, accept_sparse='csr',
                         force_all_finite=False, multi_output=True,
                         copy=True, dtype=np.float32)
        X, y = shuffle(X, y, random_state=self.random_state)

        if self._has_valid_data:
            valid_X, valid_y = check_X_y(valid_X, valid_y, accept_sparse='csr',
                                         force_all_finite=False, multi_output=True,
                                         copy=True, dtype=np.float32)

        if info is None:
            info = dict()
        datamanager = get_datamanager(X, y, info=info,
                                      feat_type=feat_type)

        name_ = datamanager.name
        self._manager = datamanager

        self.task_ = self._manager.info['task']
        self.label_num_ = self._manager.info["label_num"]
        self._time_budget = self.time_limit_total

        self._metric_with_default = self._manager.info['metric']
        self.metric_ = self._metric_with_default
        if self.mode == 'metalearning':
            self.metric_ = None
            if self.task_ in CLASSIFICATION_TASKS:
                self._metric_available = CLASSIFICATION_METRICS
            elif self.task_ in REGRESSION_TASKS:
                self._metric_available = REGRESSION_METRICS
            else:
                raise NotImplementedError()
        else:
            self._metric_available = [self.metric_]

        if self.metalearning_k <= 0:
            logging.warning("Improper value of metalearning_k, setting to 0")
            self.metalearning_k = 0

        # StopWatch setting
        self._stopwatch = StopWatch()
        self._stopwatch.start_task(name_)

        log_level = logging.INFO
        if self.debug:
            log_level = logging.DEBUG

        ###########################################################
        # setup logger                              by. bigwig
        ###########################################################
        self._logger = logging.getLogger(name_)
        formatter = logging.Formatter('[%(levelname)s] [%(asctime)s:%(name)s] %(message)s', datefmt='%H:%M:%S')

        if not self.log_filename:
            print("logging at stream")
            logHandler = logging.StreamHandler()
            logHandler.setFormatter(formatter)
        else:
            print("logging at " + self.log_filename)
            logHandler = logging.FileHandler(self.log_filename, mode='w')
            logHandler.setFormatter(formatter)

        self._logger.setLevel(log_level)
        self._logger.addHandler(logHandler)
        print("logging with " + self._logger.name)

        # ensemble checking
        if self.ensemble_size == 1:
            self._logger.info("ensemble_size=1, Non-ensemble mode.")
        elif self.ensemble_size > 1:
            self._logger.info("Now it will construct an ensemble with size %d." % self.ensemble_size)
        else:
            raise NotImplementedError("Invalid ensemble_size is given.")

        # Handling for small-size data
        if self._manager.info['train_num'] < 3000:
            if self.ensemble_size > 1:
                warnings.warn("training data is too small to construct ensemble model."
                              "ensemble_size will be set to 1", RuntimeWarning)
                self.ensemble_size = 1
            if self.resampling == 'holdout':
                warnings.warn("Now the resampling strategy is set to holdout, "
                              "but it is recommended to use cv strategy"
                              "due to its small data size", RuntimeWarning)
        elif self._manager.info['train_num'] > 100000:
            if self.resampling == 'cv':
                warnings.warn("For a large dataset, cv is inefficient compared to holdout,"
                              "while the advantage is negligible."
                              "So, we'll use holdout instead of cv.", RuntimeWarning)
                self.resampling = 'holdout'

        time_load_data = self._stopwatch.wall_elapsed(name_)
        self._print_time("LoadData", time_load_data)

        # Calculate metafeatures
        task_name = "CalculateMetafeatures"
        self._stopwatch.start_task(task_name)
        meta_features = self._manager.metafeatures(X, y)
        self._stopwatch.stop_task(task_name)
        self._print_time(task_name,
                         self._stopwatch.wall_elapsed(task_name))

        if meta_features is None:
            # Current : task is not in TASK_TYPES
            raise NotImplementedError()

        task_name = 'EncodeX'
        self._stopwatch.start_task(task_name)
        X = self._manager.encode_X(X)
        if self._has_valid_data:
            valid_X = self._manager.encode_X(valid_X, trans_only=True)
        self._stopwatch.stop_task(task_name)
        self._print_time(task_name, self._stopwatch.wall_elapsed(task_name))

        task_name = 'CalculateMetafeaturesEncoded'
        self._stopwatch.start_task(task_name)
        meta_features_enc = datamanager.metafeatures(X, y)

        if meta_features_enc is None:
            raise ValueError("meta_features_encoded is None")
        meta_features.values.update(meta_features_enc.values)

        self._stopwatch.stop_task(task_name)
        self._print_time(task_name, self._stopwatch.wall_elapsed(task_name))

        # Create a search space
        task_name = "CreateConfigSpace"
        from ConfigSpace.hyperparameters import CategoricalHyperparameter

        self._stopwatch.start_task(task_name)
        self.configuration_space = get_hyperspace(self._manager.info,
                                                  include_estimators=self.include_estimators,
                                                  include_preprocessors=self.include_preprocessors)
        #import networkx as nx
        #print(self.configuration_space.dag.node[("classifier:xgradient_boosting:max_delta_step")]["weight"])
        #import matplotlib.pyplot as plt
        #nx.draw(self.configuration_space.dag)
        n_hyperparameters = len(self.configuration_space._hyperparameters)
        idx2hps = self.configuration_space._idx_to_hyperparameter
        hps = [self.configuration_space.get_hyperparameter(idx2hps[idx])
               for idx in range(n_hyperparameters)]
        self._space_categorical = np.array(map(lambda x: isinstance(x, CategoricalHyperparameter), hps))
        self._stopwatch.stop_task(task_name)
        self._print_time(task_name,
                         self._stopwatch.wall_elapsed(task_name))

        task_name = "MetaInitialize"
        self._stopwatch.start_task(task_name)

        _task = TASK_TYPES_TO_STRING[self.task_]
        if self.task_ == MULTILABEL_CLASSIFICATION:
            _task = "binary.classification"

        cur_dir = os.path.dirname(__file__)
        meta_dir = os.path.join(cur_dir, "metalearning/files",
                                "%s_%s_%s" % (METRIC_TO_STRING[self._metric_with_default], _task,
                                              ['dense', 'sparse'][self._manager.info['is_sparse']]))

        meta_subset = (subsets['all']).copy()
        if self.task_ in CLASSIFICATION_TASKS:
            meta_subset -= EXCLUDE_META_CLASSIFICATION
        elif self.task_ in REGRESSION_TASKS:
            meta_subset -= EXCLUDE_META_REGRESSION
        meta_subset -= META_MISSING_VALUES
        meta_list = list(meta_subset)

        meta_opt = MetaLearningOptimizer(name=name_ + SENTINEL,
                                         configuration_space=self.configuration_space,
                                         meta_dir=meta_dir,
                                         metric='l1',
                                         seed=self.random_state,
                                         use_features=meta_list,
                                         subset='all',
                                         logger=self._logger)

        # TODO This is hacky, I must find a different way of adding a new dataset!
        # TODO db ? optimization point..!
        meta_opt.meta_base.add_dataset(name_ + SENTINEL, meta_features)
        runs = meta_opt.suggest_all(exclude_double_config=True)
        self.meta_initial_ = runs[:self.metalearning_k]

        self._stopwatch.stop_task(task_name)
        self._print_time(task_name, self._stopwatch.wall_elapsed(task_name))

        self.validation_score_, self.model_best_ = defaultdict(lambda: 0), {}
        self.ensemble_ = {}
        self.score_history_ = []

        task_name = "SelectModelSetting"
        self._stopwatch.start_task(task_name)

        self._dm = DumpManager(self.tmp_dir)

        mem_s = memory_usage()
        self._split_and_dump(X, y, valid_X, valid_y)
        del X, y, valid_X, valid_y
        n = gc.collect()
        self._logger.debug("Garbage Collecting... %d" % n)
        mem_data = mem_s - memory_usage()
        self._mem_expansion = 12 * mem_data

        self._stopwatch.stop_task(task_name)
        self._print_time(task_name, self._stopwatch.wall_elapsed(task_name))

        ensemble_ = self.query_model()

        task_name = "FinalizeFit"
        self._stopwatch.start_task(task_name)

        softmax_sum = 0
        for model, score in ensemble_.items():
            softmax_i = np.exp(score)
            ensemble_[model] = softmax_i
            softmax_sum += softmax_i

        for model, sm_score in ensemble_.items():
            ensemble_[model] = sm_score / softmax_sum

        for model, weight in ensemble_.items():
            estimator = joblib.load(model)
            self.ensemble_[estimator] = weight

        for m in self._metric_available:
            model = self.model_best_[m]
            estimator = joblib.load(model)
            self.model_best_[m] = estimator

        if self.mode == 'metalearning':
            for m in self._metric_available:
                self.update_metalearning(meta_features, m)
            self._logger.info("A new metalearning data is completely updated")

        self._stopwatch.stop_task(task_name)
        self._print_time(task_name, self._stopwatch.wall_elapsed(task_name))

        self._logger.info("Best model: %s" % str(self.model_best_))
        self._logger.info("Final mode: \n%s" % str(self.ensemble_))
        self._logger.info("Accuracy: %s" % self.validation_score_)

        self._logger.info("Cleaning the temporary folder")
        self._dm.clean_up()

        return self

    def predict(self, X):
        if not hasattr(self, 'ensemble_'):
            raise ValueError("Some training data should be fitted "
                             "before calling predict method")

        if self.task_ in REGRESSION_TASKS:
            X = X.copy()
            if self._manager.encoder:
                X = self._manager.encode_X(X, trans_only=True)
            predictions = []
            for model, weight in self.ensemble_.items():
                prediction = model.predict(X)
                predictions.append(prediction * weight)
            prediction = np.sum(np.array(predictions), axis=0)

        elif self.task_ in CLASSIFICATION_TASKS:
            prediction = self.predict_proba(X)
        else:
            raise NotImplementedError()

        if self.task_ == BINARY_CLASSIFICATION:
            prediction = prediction[:, 1]

        #if self.task_ in [MULTICLASS_CLASSIFICATION]:
        #    prediction = np.argmax(prediction, axis=1)
        #elif self.task_ in [BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION]:
        #    prediction = np.around(prediction)

        if len(prediction) == 0:
            raise ValueError('Something went wrong generating the predictions.')

        return prediction

    def predict_proba(self, X):
        if not hasattr(self, 'ensemble_'):
            raise ValueError("Some training data should be fitted "
                             "before calling predict_proba method")
        X = X.copy()
        if self._manager.encoder:
            X = self._manager.encode_X(X, trans_only=True)

        predictions = []
        if self.task_ in REGRESSION_TASKS:
            raise AttributeError("Regression task cannot perform predict_proba")
        for model, weight in self.ensemble_.items():
            prediction = model.predict_proba(X)
            predictions.append(prediction * weight)

        predictions = np.sum(np.array(predictions), axis=0)
        return predictions

    def score(self, X, y):
        if self.task_ in REGRESSION_TASKS:
            prediction = self.predict(X)
        else:
            prediction = self.predict_proba(X)

        return _calculate_score(y, prediction, self.task_, self.metric_)

    def query_model(self):
        from multiprocessing import Manager

        self._sync_manager = Manager()
        self._hash = {}
        defaults = get_default_configs(self._manager.info,
                                       self.configuration_space)
        initials = self.meta_initial_
        window = {}
        self.early_stopped_ = False

        task_name = "SelectModelTest"
        self._stopwatch.start_task(task_name)

        # Do not evaluate default configurations more than once
        testee = []
        for idx, config in enumerate(defaults + initials):
            if idx >= len(defaults) and config in defaults:
                continue
            testee.append(config)

        window, mfo_data, mfo_labels = self.test_configurations(testee, window)

        if self._check_early_stop():
            self._logger.info("Early-stopped")
            return window

    def test_configurations(self, configurations, window):

        results, test_hash_inv = self._evaluate(configurations)

        if len(window) == 0:
            if len(results) == 0:
                raise NoModelFittedException("Cannot find any profit model. "
                                             "Raise time_limit and just try to run again.")

            if len(results) < len(configurations) / 4:
                warnings.warn("Can't gather enough models "
                              "for this time limit setting. "
                              "Relax the time_limit parameter.", UserWarning)

        window_ = self._update_window(results, window, test_hash_inv)

        if self.validation_score_[self._metric_with_default] <= 0:
            raise NoModelFittedException("Cannot find any profit model. "
                                         "Raise time_limit and just try to run again.")

        mfo_data, mfo_labels = self._settle_test(configurations, results, self._metric_available)

        return window_, mfo_data, mfo_labels

    def _update_window(self, results, window, update_hash_inv):
        for score, model, idx in results:
            self._dm.keep_file(model)

        for score, model, idx in results:
            for m in self._metric_available:
                score_ = score[m] if self.mode == 'metalearning' else score
                if score_ > self.validation_score_[m]:
                    self.validation_score_[m] = score_
                    self.model_best_[m] = model

        best_score = self.validation_score_[self._metric_with_default]
        self.score_history_.append(best_score)

        if self.mode == "metalearning" or self.ensemble_size == 1:
            window_ = {self.model_best_[self._metric_with_default]: best_score}
            return window_

        # ensemble_size > 1 && not metalearning mode
        # In this case we updates windows as well as _hash
        raise NotImplementedError()
        for score, model in results:
            model_hash = update_hash_inv[model]

            if self._hash.has_key(model_hash):
                o_model = self._hash[model_hash]
                if score > window[o_model]:
                    del window[o_model]
                    window[model] = score
            else:
                if score > 0:
                    window[model] = score

        survived = sorted(window, key=window.get, reverse=True)[:self.ensemble_size]
        if len(window) > self.ensemble_size:
            window = {k: v for k, v in window.items() if k in survived}

        # Re-update the best model, since the issue with (unordered) dictionary
        self.model_best_[self.metric_] = survived[0]
        self.validation_score_[self.metric_] = window[survived[0]]

        return window

    def _settle_test(self, testee, results, metrics):
        mfo_data, mfo_labels = [], defaultdict(list)

        success = []
        for score, model, idx in results:
            success.append(idx)
            config = testee[idx]
            mfo_data.append(config)
            for m in metrics:
                score_ = score[m] if self.mode == "metalearning" else score
                label = 1 - max(0, score_)
                mfo_labels[m].append(label)

        fail = [id for id in range(len(testee)) if id not in success]
        for idx in fail:
            mfo_data.append(testee[idx])
            for m in metrics:
                mfo_labels[m].append(1)

        return mfo_data, mfo_labels

    def _evaluate(self, configs):
        tasks = []
        test_hash_inv = {}
        results = self._sync_manager.list()

        if self.memory_limit is None:
            self.memory_limit = self._mem_expansion + 4096
        else:
            self.memory_limit = max(3096, self.memory_limit)
        mem_limit_base = memory_usage()
        self.memory_limit += mem_limit_base

        self._logger.info("Setting memory_limit to %s" % (self.memory_limit - mem_limit_base))

        if self.n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        if self.n_jobs < 0:
            from multiprocessing import cpu_count
            self.n_jobs = max(cpu_count() + 1 + self.n_jobs, 1)

        for idx, config in enumerate(configs):
            sema = self._sync_manager.Semaphore(1)
            model_file = self._dm.alloc_file()
            test_hash_inv[model_file] = self._hash_configuration(config)

            proc = EvalProcess(config, results, sema, model_file, idx,
                               self.task_, self._datafile, self._logger,
                               self.metric_,
                               self.time_limit_each, self.memory_limit)
            tasks.append((proc, sema))

        handle_tasks(tasks, self.time_limit_each, self.memory_limit, self.n_jobs, self._logger)
        return results, test_hash_inv

    def _hash_configuration(self, configuration):
        vector = configuration._vector
        categoricals = vector[self._space_categorical]
        return hash(categoricals.tostring())

    def _split_and_dump(self, X, y, valid_X, valid_y):
        if not hasattr(self, '_dm'):
            raise ValueError("It should be called after the dumpmanager _dm is set")

        if self.resampling == 'cv':
            pass
        elif self.resampling == 'holdout':
            if not self._has_valid_data:
                data_size = y.shape[0]
                if data_size >= 100000:
                    valid_ratio = 0.3
                elif 15000 <= data_size < 100000:
                    valid_ratio = 0.2
                else:
                    valid_ratio = 0.15
                valid_size = int(data_size * valid_ratio)
                X, valid_X = X[valid_size:], X[:valid_size]
                y, valid_y = y[valid_size:], y[:valid_size]
        else:
            raise NotImplementedError()

        pkl = {"resampling": self.resampling,
               "X": X, "y": y,
               "valid_X": valid_X, "valid_y": valid_y}

        datafile = os.path.join(self._dm.dir, "data.pkl")
        joblib.dump(pkl, datafile, protocol=-1)

        self._datafile = datafile
        return datafile

    def update_metalearning(self, metafeatures, metric):
        import arff, csv
        from metalearning.aslib_simple import AlgorithmSelectionProblem

        cur_dir = os.path.dirname(__file__)
        meta_dir = os.path.join(cur_dir, "metalearning/files",
                                "%s_%s_%s" % (METRIC_TO_STRING[metric],
                                              TASK_TYPES_TO_STRING[self.task_],
                                              ['dense', 'sparse'][self._manager.info["is_sparse"]]))

        meta_reader = AlgorithmSelectionProblem(meta_dir, self._logger)
        config_info = meta_reader.get_config_info()

        config = self.model_best_[metric].configuration

        with open(os.path.join(meta_dir, 'configurations.csv'), 'a') as f:
            wrt = csv.DictWriter(f, config_info['fields'])
            new_row = {}
            new_idx = int(config_info['last_idx']) + 1
            new_row['idx'] = new_idx
            for key in config_info['fields']:
                if key == 'idx':
                    continue
                new_row[key] = config[key]
            wrt.writerow(new_row)

        with open(os.path.join(meta_dir, 'algorithm_runs.arff'), 'r') as f:
            arff_dict = arff.load(f)
            data = []
            for key, type_ in arff_dict['attributes']:
                if key.upper() == 'INSTANCE_ID':
                    data.append(self._manager.name)
                elif key.upper() == "REPETITION":
                    data.append(1)
                elif key.upper() == "ALGORITHM":
                    data.append(new_idx)
                elif key.upper() == "RUNSTATUS":
                    data.append('ok')
                else:
                    val = self.validation_score_[metric]
                    data.append(val)
            arff_dict['data'].append(data)
        with open(os.path.join(meta_dir, 'algorithm_runs.arff'), 'w') as f:
            arff.dump(arff_dict, f)

        with open(os.path.join(meta_dir, 'feature_values.arff'), 'r') as f:
            arff_dict = arff.load(f)
            data = []
            for key, type_ in arff_dict['attributes']:
                if key.upper() == 'INSTANCE_ID':
                    data.append(self._manager.name)
                elif key.upper() == "REPETITION":
                    data.append(1)
                else:
                    try:
                        val = metafeatures.values[key].value
                    except KeyError:
                        val = '?'
                    data.append(val)
            arff_dict['data'].append(data)
        with open(os.path.join(meta_dir, 'feature_values.arff'), 'w') as f:
            arff.dump(arff_dict, f)

    def _check_early_stop(self):
        score = self.validation_score_[self._metric_with_default]
        has_eps_loss = score > 1 - SCORE_EPSILON
        if has_eps_loss:
            self._logger.info("Found epsilon-loss model")
            return True

        if self.mode == 'alice':
            self._logger.info("Alice mode")
            return True

    def _print_time(self, task_name, time_check):
        self._time_budget = max(0, self._time_budget - time_check)
        self._logger.info('%s | remaining time: %5.2f sec | memory usage: %5.2f MB' %
                          (task_name, self._time_budget, memory_usage()))
