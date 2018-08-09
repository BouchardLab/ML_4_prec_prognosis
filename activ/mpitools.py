from mpi4py import MPI
import logging
from os.path import abspath

class MPIFileHandler(logging.FileHandler):
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND,
                 encoding='utf-8',
                 delay=False,
                 comm=MPI.COMM_WORLD ):
        self.baseFilename = abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
           logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open( self.comm, self.baseFilename, self.mode )
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.

        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None

