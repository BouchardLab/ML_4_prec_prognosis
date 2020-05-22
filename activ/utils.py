import sys
import logging
import math
from sklearn.preprocessing import LabelEncoder
import time

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


def check_seed(x):
    if len(x) == 0:
        return int(round(time.time() * 1000)) % (2**32 - 1)
    try:
        return int(x)
    except Exception as e:
        raise argparse.ArgumentTypeError(e.args[0])


def int_list(string):
    if ':' in string:
        ar = [int(a) for a in string.split(":")]
        return list(range(ar[0], ar[1]+1))
    else:
        return list(map(int, string.split(",")))
