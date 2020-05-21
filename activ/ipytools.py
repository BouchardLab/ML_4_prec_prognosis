# This code can be put in any Python module, it does not require IPython
# itself to be running already.  It only creates the magics subclass but
# doesn't instantiate it yet.
from __future__ import print_function
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)

import traceback as tb
import logging
import os
import sys


# The class MUST call this class decorator at creation time
@magics_class
class LoggerMagic(Magics):
    def __init__(self, shell):
        super(LoggerMagic, self).__init__(shell)

    @line_magic
    def lmagic(self, line):
        "my line magic"
        print("Full access to the main IPython object:", self.shell)
        print("Variables in the user namespace:", list(self.shell.user_ns.keys()))
        return line

    @cell_magic
    def logexc(self, line, cell):
        "my cell magic for logging"
        args = line.split()
        if len(args) == 0:
            logger_varname = 'logger'
        else:
            logger_varname = args[0]
        logger = self.shell.user_ns.get(logger_varname)
        if not isinstance(logger, logging.Logger):
            sys.stderr("Cannot find Logger variable '%s'" % logger_varname)
            return line, cell
        try:
            self.shell.ex(cell)
        except Exception as e:
            logger.info("caugh exception:\n%s" % tb.format_exc())
            raise e

    @line_cell_magic
    def lcmagic(self, line, cell=None):
        "Magic that works both as %lcmagic and as %%lcmagic"
        if cell is None:
            print("Called as line magic")
            return line
        else:
            print("Called as cell magic")
            return line, cell

def get_logger(path=None, samedir=None, subdir=None, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    formatter = logging.Formatter(fmt)
    logger = logging.getLogger()
    if path is not None:
        if samedir is not None:
            if isinstance(subdir, list):
                subdir.append(path)
            else:
                subdir = list(os.path.split(subdir)) + [path]
            path = get_samedir(samedir, *subdir)
        else:
            if isinstance(subdir, list):
                subdir.append(path)
            else:
                subdir = os.path.split(subdir) + [path]
            path = os.path.join(subdir)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def get_samedir(path, *filename):
    parts = [os.path.dirname(path)] + list(filename)
    ret = os.path.join(*parts)
    outdir = os.path.dirname(ret)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return ret
