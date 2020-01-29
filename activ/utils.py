import sys
import logging
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

def get_logger(path=None, name=None, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    formatter = logging.Formatter(fmt)
    logger = logging.Logger(name)
    if path is not None:
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def read_labels(dset):
    classes = dset.attrs['classes'].astype('U')
    enc = LabelEncoder().fit(classes)
    return enc.inverse_transform(dset[:]), classes

def get_start_portion(rank, size, n):
    portion = math.ceil((n - rank) / size)
    fbsize = math.ceil(n/size)
    fmr = size - fbsize*size + n
    start = fbsize*fmr + portion * (rank - fmr)
    return start, portion
