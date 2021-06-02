import time


class Timer(object):
    def __init__(self):
        self._tic = time.time()

    def tic(self):
        self._tic = time.time()
        return self._tic

    def toc(self):
        return time.time() - self._tic

    def tictoc(self):
        _toc = time.time()
        duration = _toc - self._tic
        self._tic = _toc
        return duration
