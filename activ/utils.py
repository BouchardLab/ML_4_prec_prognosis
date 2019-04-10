import sys
import logging
import math


def get_logger(logger_name, comm=None, path=None, quiet=False):
    rank, size = 1, 1
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
    logfmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if size > 1:
        from activ.mpitools import MPILogHandler
        logger_name = logger_name + "-%d" % rank
        if path is None:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = MPILogHandler(path, comm)
    else:
        stream = sys.stdout if path is None else open(path, 'w')
        handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(logfmt))
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)

    log_level = logging.INFO
    if quiet:
        log_level = logging.WARNING
    logger.setLevel(log_level)
    return logger


def get_start_portion(rank, size, n):
    portion = math.ceil((n - rank) / size)
    fbsize = math.ceil(n/size)
    fmr = size - fbsize*size + n
    start = fbsize*fmr + portion * (rank - fmr)
    return start, portion
