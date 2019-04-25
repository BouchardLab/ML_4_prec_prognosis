import abc as _abc
import numpy as _np

class AbstractSampler(object, metaclass=_abc.ABCMeta):

    def _check_Xs(self, *arrays):
        for i in range(1, len(arrays)):
            if arrays[i].shape[0] != arrays[i-1].shape[0]:
                raise ValueError('arrays must have the same number of samples')

    @_abc.abstractmethod
    def sample(self, *arrays):
        """
        Return an generator over samples
        """
        pass

    @_abc.abstractmethod
    def get_n_iters(self, *arrays):
        """
        The number of iterations produced by *sample*
        """
        pass

    @_abc.abstractmethod
    def get_n_samples(self, *arrays):
        """
        The number of samples produced at each iteration
        """
        pass


class SubSampler(AbstractSampler):

    def __init__(self, n_iters, subsample_size=1.0, random_state=None):
        self.random_state = check_random_state(random_state)
        self.subsample_size = subsample_size
        self.n_iters = n_iters

    def sample(self, *arrays):
        ss_n = self._get_ssn(*arrays)
        n = arrays[0].shape[0]
        for i in range(self.n_iters):
            idx = self.random_state.permutation(n)[:ss_n]
            if len(arrays) == 1:
                yield arrays[0][idx]
            else:
                yield tuple(dset[idx] for dset in arrays)

    def _get_ssn(self, *arrays):
        self._check_Xs(*arrays)
        if isinstance(self.subsample_size, (int, _np.int32, _np.int16, _np.int8)):
            return self.subsample_size
        else:
            self._check_Xs(*arrays)
            n = arrays[0].shape[0]
            return int(self.subsample_size * n)

    def get_n_iters(self, *arrays):
        return self.n_iters

    def get_n_samples(self, *arrays):
        return self._get_ssn(*arrays)


class BootstrapSampler(AbstractSampler):

    def __init__(self, n_iters, random_state=None):
        self.random_state = check_random_state(random_state)
        self.n_iters = n_iters

    def get_n_iters(self, *arrays):
        return self.n_iters

    def sample(self, *arrays):
        self._check_Xs(*arrays)
        n = arrays[0].shape[0]
        for i in range(n):
            idx = self.random_state.randint(n, size=n)
            if len(arrays) == 1:
                yield arrays[0][idx]
            else:
                yield tuple(dset[idx] for dset in arrays)

    def get_n_samples(self, *arrays):
        self._check_Xs(*arrays)
        return arrays[0].shape[0]


class JackknifeSampler(AbstractSampler):

    def __init__(self, indices=None):
        if isinstance(indices, (tuple, list)):
            self.indices = _np.array(indices)
        else:
            self.indices = indices

    def get_n_iters(self, *arrays):
        self._check_Xs(*arrays)
        if self.indices is None:
            return arrays[0].shape[0]
        else:
            return self.indices.shape[0]

    def sample(self, *arrays):
        self._check_Xs(*arrays)
        n = arrays[0].shape[0]
        indices = self.indices
        if indices is None:
            indices = _np.arange(n)
        mask = _np.ones(n, dtype=bool)
        ar_idx = _np.arange(n)
        for i in indices:
            mask[i] = False
            idx = ar_idx[mask]
            mask[i] = True
            if len(arrays) == 1:
                yield arrays[0][idx]
            else:
                yield tuple(dset[idx] for dset in arrays)

    def get_n_samples(self, *arrays):
        self._check_Xs(*arrays)
        return arrays[0].shape[0] - 1


