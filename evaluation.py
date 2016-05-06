import resource
import time
from collections import defaultdict

import numpy as np
from sklearn import cross_validation
from multiprocessing import Process
import joblib

from constants import *
from metrics import classification_metrics
from metrics import regression_metrics
from metrics.util import sanitize_array, \
    normalize_array
from utils.exceptions import *
from pipeline.utils import retrieve_template

def _calculate_score(solution, prediction, task_type, metric=None):

    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    solution = np.array(solution, dtype=np.float32)

    if task_type == MULTICLASS_CLASSIFICATION:
        # This used to crash on travis-ci; special treatment to find out why
        # it crashed!
        solution_binary = np.zeros(prediction.shape)

        for i in range(solution_binary.shape[0]):
            label = int(np.round_(solution[i]))
            solution_binary[i, label] = 1
        solution = solution_binary

    elif task_type == BINARY_CLASSIFICATION:
        solution = solution.reshape(-1, 1)
        prediction = prediction[:, 1].reshape(-1, 1)

    if solution.shape != prediction.shape:
        raise ValueError("Solution shape %s != prediction shape %s" %
                         (solution.shape, prediction.shape))

    if metric is None:
        score = dict()
        if task_type in REGRESSION_TASKS:
            cprediction = sanitize_array(prediction)
            for metric_ in REGRESSION_METRICS:
                score[metric_] = regression_metrics.calculate_score(metric_,
                                                                    solution,
                                                                    cprediction)
        else:
            csolution, cprediction = normalize_array(solution, prediction)
            for metric_ in CLASSIFICATION_METRICS:
                score[metric_] = classification_metrics.calculate_score(
                    metric_, csolution, cprediction, task_type)

        for metric_ in score:
            if np.isnan(score[metric_]):
                score[metric_] = 0


    else:
        if task_type in REGRESSION_TASKS:
            cprediction = sanitize_array(prediction)
            score = regression_metrics.calculate_score(metric,
                                                       solution,
                                                       cprediction)
        else:
            csolution, cprediction = normalize_array(solution, prediction)
            score = classification_metrics.calculate_score(metric,
                                                           csolution,
                                                           cprediction,
                                                           task=task_type)
        if np.isnan(score):
            score = 0

    return score


def evaluate_estimator(datafile, estimator, task,
                       metric=None,
                       logger=None):
    if metric and metric not in METRIC:
        raise ValueError("Invalid metric")

    def scorer(estimator, X, y):
        if task in REGRESSION_TASKS:
            y_pr = estimator.predict(X)
        elif task in CLASSIFICATION_TASKS:
            y_pr = estimator.predict_proba(X, batch_size=1000)
        else:
            raise NotImplementedError()
        score = _calculate_score(y, y_pr, task, metric)

        return score

    eval_s = time.time()

    data_pkl = joblib.load(datafile, 'r')
    resampling = data_pkl['resampling']
    if resampling == 'holdout':
        X_tr = data_pkl["X"]
        y_tr = data_pkl["y"]
        X_val = data_pkl["valid_X"]
        y_val = data_pkl["valid_y"]
        estimator.fit(X_tr, y_tr)
        score = scorer(estimator, X_val, y_val)
    elif resampling == 'cv':
        X, y = data_pkl["X"], data_pkl["y"]
        cv = cross_validation.check_cv(None, X, y, classifier=(task in CLASSIFICATION_TASKS))

        score = defaultdict(list) if metric is None else []
        for train, test in cv:
            X_tr, X_val = X[train], X[test]
            y_tr, y_val = y[train], y[test]
            estimator.fit(X_tr, y_tr)
            score_ = scorer(estimator, X_val, y_val)
            if metric is None:
                for m in score_:
                    score[m].append(score_[m])
            else:
                score.append(score_)
        if metric is None:
            for m in score:
                score[m] = np.mean(score[m])
        else:
            score = np.mean(score)
        estimator.fit(X, y)
    else:
        raise NotImplementedError()

    eval_e = time.time()
    if logger:
        logger.debug("Evaluation done, score: %s | %s sec\n%s" % (score, eval_e-eval_s, estimator))

    return score


class EvalProcess(Process):
    def __init__(self, configuration, result, sema, model_file, id,
                 task, datafile, logger,
                 metric=None, time_limit=None, memory_limit=None):
        super(EvalProcess, self).__init__()
        self.id = id
        self.configuration = configuration
        self.task = task
        self.datafile = datafile
        self.logger = logger
        self.metric = metric
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.result = result
        self.sema = sema
        self.model_file = model_file

        # Set the process as a demonic
        self._daemonic = True

    def run(self):
        self.logger.debug("Test configuration %s" % self.configuration)
        signal.signal(signal.SIGALRM, handler)
        signal.signal(signal.SIGXCPU, handler)
        signal.signal(signal.SIGQUIT, handler)
        # set the memory limit
        if self.memory_limit is not None:
            # byte --> megabyte
            mem_in_b = self.memory_limit * 1024 * 1024
            # the maximum area (in bytes) of address space which may be taken by the process.
            resource.setrlimit(resource.RLIMIT_AS, (mem_in_b, -1))

        if self.time_limit is not None:
            # From the Linux man page:
            # When the process reaches the soft limit, it is sent a SIGXCPU signal.
            # The default action for this signal is to terminate the process.
            # However, the signal can be caught, and the handler can return control
            # to the main program. If the process continues to consume CPU time,
            # it will be sent SIGXCPU once per second until the hard limit is reached,
            # at which time it is sent SIGKILL.
            resource.setrlimit(resource.RLIMIT_CPU, (self.time_limit, -1))

        estimator = retrieve_template(self.task, self.configuration)

        try:
            score = evaluate_estimator(self.datafile, estimator,
                                       self.task, self.metric,
                                       self.logger)
        except MemoryError:
            self.logger.debug("MemoryError")
            return

        except OSError as e:
            if (e.errno == 11):
                self.logger.debug("SubprocessException")
            else:
                self.logger.debug("AnythingException")
            return

        except CpuTimeoutException:
            self.logger.debug("CpuTimeoutException")
            return

        except TimeoutException:
            self.logger.debug("TimeoutException")
            return

        except AnythingException as e:
            self.logger.debug("AnythingException")
            return
        except:
            self.logger.debug("Some wired exception occurred")
            raise

        resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
        resource.setrlimit(resource.RLIMIT_CPU, (-1, -1))

        self.sema.acquire()
        joblib.dump(estimator, self.model_file, protocol=-1)
        self.result.append((score, self.model_file, self.id))
        self.sema.release()
