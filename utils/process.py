import psutil
from signal import SIGKILL
import os
import time

def memory_usage(pid=None):
    # return the memory usage in MB
    if not pid:
        pid = os.getpid()
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return -1
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def handle_tasks(tasks, time_limit, memory_limit, n_jobs=1, logger=None):
    window = []
    while len(tasks) > 0:
        while len(window) < n_jobs:
            if len(tasks) == 0:
                break
            job, sema = tasks.pop()
            window.append((job, time.time(), sema))
            job.start()
            if logger:
                logger.debug("Process %s start: %s MB is used" % (job.pid, memory_usage()))

        while len(window) == n_jobs:
            window = _check_window(window, time_limit, memory_limit, logger)

    while len(window) > 0:
        window = _check_window(window, time_limit, memory_limit, logger)


def _check_window(window, time_limit, memory_limit, logger=None):
    for job, start, sema in window:
        if time.time() - start > time_limit:
            sema.acquire()
            if job.is_alive():
                if logger:
                    logger.debug("Process %s will be killed: "
                                 "Resource usage violation" % job.pid)
                os.kill(job.pid, SIGKILL)
            sema.release()
    time.sleep(0.1)
    return [j for j in window if j[0].is_alive()]
