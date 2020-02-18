"""
Copula construction

Author
------
Erwin de Gelder

Creation
--------
17 Feb 2020


To do
-----


Modifications
-------------
"""

from typing import List
import numpy as np
import scipy.special
from fastkde import KDE, process_reshaped_data
from options import Options


class CopulaOptions(Options):
    """ Options for the copula. """


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
    kdes: List[KDE] = []


class Copula:
    """ Constructing the probability density function using a copula.

    Attributes:
        data: The data that is used to construct the copula.
    """
    def __init__(self, data: np.ndarray, options: CopulaOptions = None):
        self.data = data
        self.options = CopulaOptions() if options is None else options
        self.parms = CopulaParameters()

        self.parms.dim = self.data.shape[1]

    def fit(self) -> None:
        """ Fit """
        # Construct KDEs of marginal probabilities.
        self.parms.kdes = [KDE(self.data[:, i]) for i in range(self.parms.dim)]
        for kde in self.parms.kdes:
            kde.compute_bandwidth()

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
