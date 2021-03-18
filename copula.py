""" Copula construction

Creation date: 2020 02 17
Author(s): Erwin de Gelder

Modifications:
2020 02 19: Add class for copula pairs.
"""

from copy import deepcopy
from typing import List, Sequence, Tuple
import numpy as np
import scipy.special
from .fastkde import KDE, process_reshaped_data
from .options import Options


class CopulaOptions(Options):
    """ Options for the copula. """
    def __init__(self, **kwargs):
        self.kdes_bandwidths: List[float] = []
        Options.__init__(self, **kwargs)


class CopulaParameters(Options):
    """ Parameters of the copula.

    This class contains the following attributes:
        data_uniform: Data converted to be normally distributed.
        data_uniform: Data converted to be in the range [0, 1].
        dim: Dimension of the data.
        kde_copula: KDE of the copula.
        kdes: The univariate kernel density estimations.
    """
    data_normal: np.ndarray = np.array([])
    data_uniform: np.ndarray = np.array([])
    dim: int = 0
    kde_copula: KDE = None

    def __init__(self):
        self.kdes: List[KDE] = []
        Options.__init__(self)


class Copula:
    """ Constructing the probability density function using a copula.

    Attributes:
        data: The data that is used to construct the copula.
        options: Options for the copula, see CopulaOptions.
        parms: Parameters of the copula, see CopulaParameters.
    """
    def __init__(self, data: np.ndarray, options: CopulaOptions = None):
        self.data = data
        self.options = CopulaOptions() if options is None else options
        self.parms = CopulaParameters()

        self.parms.dim = self.data.shape[1]
        if not self.options.kdes_bandwidths:
            self.options.kdes_bandwidths = [None for _ in range(self.parms.dim)]
        self.fit()

    def fit(self) -> None:
        """ Fit """
        # Construct KDEs of marginal probabilities.
        self.parms.kdes = [KDE(self.data[:, i]) for i in range(self.parms.dim)]
        for kde, bandwidth in zip(self.parms.kdes, self.options.kdes_bandwidths):
            if bandwidth is None:
                kde.compute_bandwidth()
            else:
                kde.set_bandwidth(bandwidth)

        # Convert the data such that is in the range [0, 1].
        self.parms.data_uniform = np.array(np.zeros_like(self.data))
        for i, kde in enumerate(self.parms.kdes):
            self.parms.data_uniform[:, i] = kde.cdf(self.data[:, i])

        # Convert data such that each marginal is normally distributed.
        self.parms.data_normal = scipy.special.erfinv(self.parms.data_uniform*2-1)*np.sqrt(2)

        # Construct the KDE of the "normally" distributed data.
        self.parms.kde_copula = KDE(self.parms.data_normal)
        self.parms.kde_copula.compute_bandwidth()

    def pdf(self, xdata: np.ndarray) -> np.ndarray:
        """ Compute the probability density function at the given points.

        :param xdata: The points for which the probability density is required.
        :return: The probability density.
        """
        xdata = xdata.copy()
        return process_reshaped_data(xdata, self._pdf)

    def _pdf(self, xdata: np.ndarray) -> np.ndarray:
        data_uniform = np.array(np.zeros_like(xdata))
        for i, kde in enumerate(self.parms.kdes):
            data_uniform[:, i] = kde.cdf(xdata[:, i])
        data_normal = scipy.special.erfinv(data_uniform*2-1)*np.sqrt(2)

        # Compute the log pdf of the copula.
        scores = self.parms.kde_copula.score_samples(data_normal)
        scores *= (2*np.pi)**(self.parms.dim/2)
        scores /= np.exp(-np.sum(data_normal**2, axis=1)/2)

        # Add the log pdf of the marginals.
        for i, kde in enumerate(self.parms.kdes):
            scores *= kde.score_samples(xdata[:, i])
        return scores

    def copula(self, xdata: np.ndarray) -> np.ndarray:
        """ Compute the actual copula, i.e., the cdf on the interval [0,1]^d.

        :param xdata: The points for which the copula is required.
        :return: The copula values.
        """
        return process_reshaped_data(xdata, self._copula)

    def _copula(self, xdata: np.ndarray) -> np.ndarray:
        xdata = scipy.special.erfinv(xdata*2-1)*np.sqrt(2)
        return self.parms.kde_copula.cdf(xdata)

    def cdf(self, xdata: np.ndarray) -> np.ndarray:
        """ Compute the actual cumulative distribution.

        :param xdata: The points for which the cdf is required.
        :return: The cdf values.
        """
        return process_reshaped_data(xdata, self._cdf)

    def _cdf(self, xdata: np.ndarray) -> np.ndarray:
        data_uniform = np.array(np.zeros_like(xdata))
        for i, kde in enumerate(self.parms.kdes):
            data_uniform[:, i] = kde.cdf(xdata[:, i])
        return self._copula(data_uniform)


class CopulaPairsParameters(Options):
    """ Parameters of the copula pairs.

    This class contains the following attributes:
        common_vars: List of the common variables.
    """
    def __init__(self):
        self.common_vars: List[Tuple[int, int]] = []
        self.copulas: List[Copula] = []
        Options.__init__(self)


class CopulaPairs:
    """ Model the distribution with multiple pair copulas.

    Attributes:
        data: The data that is used to construct the copula.
        pairs: The pairs of the copula.
        options: Options for the copula, see CopulaOptions.
        parms: Parameters of the copula pairs, see CopulaPairsParameters
    """
    def __init__(self, data: np.ndarray, pairs: Sequence[Tuple[int, int]],
                 options: CopulaOptions = None):
        self.data = data
        if len(pairs) != (data.shape[1] - 1):
            raise ValueError("Number of pairs must be the number of dimensions minus one.")
        self.pairs = pairs
        self.options = CopulaOptions() if options is None else options
        self.parms = CopulaPairsParameters()
        self.fit()

    def fit(self) -> None:
        """ Fit the copulas. """
        self.parms.common_vars = []
        for i, (parta, partb) in enumerate(self.pairs):
            # Loop through the previous pairs and see if we have seen these variables earlier.
            bandwidths = [None, None]
            for copula, pair in zip(self.parms.copulas[:i], self.pairs[:i]):
                if parta in pair:
                    self.parms.common_vars.append((i, 0))
                    bandwidths[0] = copula.parms.kdes[pair.index(parta)].get_bandwidth()
                if partb in pair:
                    self.parms.common_vars.append((i, 1))
                    bandwidths[1] = copula.parms.kdes[pair.index(partb)].get_bandwidth()

            # Construct the copula.
            options = deepcopy(self.options)
            if not options.kdes_bandwidths:
                options.kdes_bandwidths = bandwidths
            self.parms.copulas.append(Copula(self.data[:, [parta, partb]], options=options))

    def pdf(self, xdata: np.ndarray) -> np.ndarray:
        """ Compute the probability density function at the given points.

        :param xdata: The points for which the probability density is required.
        :return: The probability density.
        """
        return process_reshaped_data(xdata, self._pdf)

    def _pdf(self, xdata: np.ndarray) -> np.ndarray:
        score = np.ones(len(xdata))
        for copula, pair in zip(self.parms.copulas, self.pairs):
            score *= copula.pdf(xdata[:, pair])
        for icopula, ivar in self.parms.common_vars:
            score /= self.parms.copulas[icopula].parms.kdes[ivar].score_samples(
                xdata[:, self.pairs[icopula][ivar]])
        return score
