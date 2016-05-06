import signal

class CpuTimeoutException(Exception):
    pass


class TimeoutException(Exception):
    pass


class MemoryLimitException(Exception):
    pass


class SubprocessException(Exception):
    pass


class AnythingException(Exception):
    pass


def handler(signum, frame):
    # logs message with level debug on this logger
    if signum == signal.SIGXCPU:
        # when process reaches soft limit --> a SIGXCPU signal is sent
        # (it normally terminats the process)
        raise CpuTimeoutException
    elif signum == signal.SIGALRM:
        # SIGALRM is sent to process when the specified time limit to an alarm function elapses
        # (when real or clock time elapses)
        raise TimeoutException
    raise AnythingException