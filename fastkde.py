""" Kernel Density Estimation

Creation date: 2018 07 18
Author(s): Erwin de Gelder

Modifications:
2018 08 08 Add functionality for adding data.
2018 08 10 Fixed mistakes for computing one-leave-out score.
2018 09 27 Several improvements. Score now computed with only the first n datapoints. Speed
           improved.
2018 10 01 Some comments added.
2018 10 09 Added computation of gradient and laplacian of the KDE.
2018 10 13 Changed computation of scores and laplacian such that less memory is used.
2018 11 06 Improve PEP8 compliancy.
2018 11 07 Add description of class.
2019 08 30 Change type hinting: np.array should be np.ndarray.
2020 02 14 Add function for computing the cumulative distribution function.
2020 02 17 Use special classes instead of dictionaries.
2020 02 21 Add the possibility to have a variable bandwidth.
2020 03 06 Return warning number when computing the bandwidth.
2020 05 01 Add the option of normalizing the data.
"""

import time
from typing import Callable, Union
import numpy as np
import scipy.spatial.distance as dist
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
from .options import Options


class KDEConstants(Options):
    """ Constants that are used for the various methods.

    The following constants are included:
        ndata(int): Number of datapoints that are used (can be smaller than the number of
                    datapoints in the data attribute.
        const_score(float): Constant part of leave-one-out score.
        dim(float): Dimension of data.
        muk(float): integral [ kernel(x)^2 ]. Since Gaussian kernel is used: 1/(2pi)^(d/2).
        invgr(float): Inverse of Golden Ratio (used for Golden Section Search).
        invgr2(float): Inverse of squared Golden Ratio (used for Golden Section Search).
        variable_bandwidth(bool): Whether a variable bandwidth is used.
    """
    ndata: int = 0
    const_score: float = 0
    dim: int = 0
    muk: float = 0
    invgr: float = (np.sqrt(5) - 1) / 2
    invgr2: float = (3 - np.sqrt(5)) / 2
    variable_bandwidth: bool = False
    percentile: float = 95
    bandwidth_factor: float = 1


class KDEData(Options):
    """ Several np.ndarrays that are used for the various methods.

    The following variables are included:
        mindists(np.ndarray): Negative (minus) of euclidean distances.
        data_score_samples(np.ndarray): Scores of each sample of data.
        newshape(np.ndarray): The new shape of the data to be returned.
        data_dist(np.ndarray): Euclidean distance of KDE data and input data.
        difference(np.ndarray): Difference of KDE data and input data.
        std(np.ndarray): Standard deviation of data, only calculated if data
            must be scaled.
    """
    mindists: np.ndarray = np.array([])
    data_score_samples: np.ndarray = np.array([])
    newshape: np.ndarray = np.array([])
    data_dist: np.ndarray = np.array([])
    difference: np.ndarray = np.array([])
    std: np.ndarray = np.array([])
    # self.xhist, self.yhist, self.fft = None, None, None  # Not used at the moment


class KDE:
    """ Kernel Density Estimation

    This class can be utilized to create Kernel Density Estimations (KDE) of data.

    Inference of the KDE is significantly faster than existing KDE tools from, e.g.,
    sklearn and scipy. This is especially the case when one needs to infer at the same
    datapoints (e.g., each time with a different bandwidth), because the squared euclidean
    distance between the points at which  the KDE needs to be evaluated and the points that are
    used to construct the KDE is only calculated once (if the method set_score_samples() is used).
    Also the leave-one-out cross validation is significantly faster than existing KDE tools,
    because of the same reason: the euclidean distance between the datapoints is only calculated
    once.

    Attributes:
        bandwidth(float): The bandwidth of the KDE. If no bandwidth is set or computed,
            the bandwidth equals None.
        data(np.ndarray): The data that is used to construct the KDE.
        constants(KDEConstants): Constants that are used for the various methods.
        data_helpers(KDEData): Several np.ndarrays that are used for the various methods.
    """
    def __init__(self, data: np.ndarray = None, bandwidth: float = None, scaling: bool = False):
        self.bandwidth = bandwidth
        self.data = None
        self.constants = KDEConstants()
        self.data_helpers = KDEData()
        self.scaling = scaling
        self.fit(data)

    def fit(self, data: np.ndarray) -> None:
        """ Fit the data

        The data is stored. Furthermore, some calculations are done:
         - Computing the dimension of the data.
         - Computing the corresponding constant muk = integral[ kernel(x)^2 ]

        :param data: The provided data. When multidimensional data is used, the data should be
            an n-by-d array, with n datapoints and d dimensions.
        """
        if len(data.shape) == 1:
            self.data = data[:, np.newaxis]
        else:
            self.data = data

        if self.scaling:
            self.data_helpers.std = np.std(self.data, axis=0)
            self.data /= self.data_helpers.std

        # Note: creating the distance matrix takes quite some time and is only needed if cross
        # validation is performed.
        # Therefore, this is not done here. Only the first time when cross validation is performed
        self.constants.dim = self.data.shape[1]
        self.set_n(len(self.data))
        # muk = integral[ kernel(x)^2 ]
        self.constants.muk = 1 / (2 ** self.constants.dim * np.sqrt(np.pi ** self.constants.dim))

        # # Build histogram (for using FFT for computing bandwidth)
        # sigma = np.std(self.data)
        # self.yhist, bin_edges = np.histogram(self.data,
        #                                      int((np.max(self.data) - np.min(self.data) /
        #                                           (sigma/100))),
        #                                      range=(np.min(self.data) - 3*sigma,
        #                                             np.max(self.data) + 3*sigma),
        #                                      density=True)
        # self.xhist = (bin_edges[:-1] + bin_edges[1:]) / 2
        # self.fft = np.fft.fft(self.yhist)

    def set_n(self, ndatapoints: int) -> None:
        """ Set the number of datapoints to be used when evaluating the one-leave-out score.

        The constant term of the score for the one-leave-out cross validation is set.

        :param ndatapoints: Number of datapoints
        """
        self.constants.ndata = ndatapoints
        self.constants.const_score = (-ndatapoints*self.constants.dim/2 * np.log(2 * np.pi) -
                                      ndatapoints*np.log(ndatapoints-1))

    def add_data(self, newdata: np.ndarray) -> None:
        """ Add extra data

        The extra data is stored. The number of datapoints is updated. In case the matrix with
        euclidean distances is already computed, this matrix will be updated as well.

        :param newdata: The provided data. When multidimensional data is used, the data should be
            an n-by-d array, with n datapoints and d dimensions.
        """
        if len(newdata.shape) == 1:
            newdata = newdata[:, np.newaxis]
        if self.scaling:
            newdata /= self.data_helpers.std
        nnew = len(newdata)

        # Expand the matrix with the distances if this matrix is already defined
        if self.data_helpers.mindists is not None:
            newmindists = dist.squareform(dist.pdist(newdata, metric='sqeuclidean')) / 2
            newmindists *= -1  # Do it this way in order to prevent invalid warning
            oldmindists = -dist.cdist(self.data, newdata, metric='sqeuclidean') / 2
            self.data_helpers.mindists = \
                np.concatenate((np.concatenate((self.data_helpers.mindists, oldmindists),
                                               axis=1),
                                np.concatenate((np.transpose(oldmindists), newmindists), axis=1)),
                               axis=0)

        # Update other stuff
        self.data = np.concatenate((self.data, newdata), axis=0)
        self.set_n(self.constants.ndata + nnew)

    def compute_bandwidth(self, **kwargs) -> int:
        """ Compute the bandwidth

        Currently, the Golden Section Search is used for this. See compute_bandwidth_gss for
        more details.

        :return: An error/warning integer.
        """
        return self.compute_bandwidth_gss(**kwargs)

    def compute_bandwidth_grid(self, min_bw: float = 0.001, max_bw: float = 1.0,
                               n_bw: int = 200) -> None:
        """ Grid search for optimal bandwidth

        :param min_bw: The minimum bandwidth to look for.
        :param max_bw: The maximum bandwidth to look for.
        :param n_bw: The number of bandwidths to look for.
        """
        bandwidths = np.linspace(min_bw, max_bw, n_bw)
        score = np.array(np.zeros_like(bandwidths))
        for i, bandwidth in enumerate(bandwidths):
            score[i] = self.score_leave_one_out(bandwidth=bandwidth)
        self.bandwidth = bandwidths[np.argmax(score)]

    def compute_bandwidth_gss(self, min_bw: float = 0.001, max_bw: float = 1., max_iter: int = 100,
                              tol: float = 1e-5) -> int:
        """ Golden section search.

        Given a function f with a single local minimum in
        the interval [a,b], gss returns a subset interval
        [c,d] that contains the minimum with d-c <= tol.

        Meaning of the returned integer:
         - 0: no warning/error
         - 1: searched only on the right side, so max_bw might need to be larger
         - 2: searched only on the left side, so min_bw might need to be smaller

        :param min_bw: The minimum bandwidth to look for.
        :param max_bw: The maximum bandwidth to look for.
        :param max_iter: The maximum number of iterations to perform.
        :param tol: The tolerance that determines when the algorithm is terminated.
        :return: An warning/error integer.
        """
        difference = max_bw - min_bw
        datapoints = np.array([min_bw, 0, 0, max_bw])
        datapoints[1] = datapoints[0] + self.constants.invgr2 * difference
        datapoints[2] = datapoints[0] + self.constants.invgr * difference

        # required steps to achieve tolerance
        n_iter = int(np.ceil(np.log(tol / difference) / np.log(self.constants.invgr)))
        n_iter = max(1, min(n_iter, max_iter))

        bandwidth_normalized = []
        if not self.constants.variable_bandwidth:
            scores = [self.score_leave_one_out(bandwidth=datapoints[1]),
                      self.score_leave_one_out(bandwidth=datapoints[2])]
        else:
            # Setup bandwidth in case it is variable
            # Note that mindists need to be defined. To be sure of that, we just run
            # score_leave_one_out() once.
            if not self.data_helpers.mindists.size:
                self.score_leave_one_out(bandwidth=np.ones(len(self.data)))
            bandwidth_normalized = np.sqrt(-2*np.percentile(self.data_helpers.mindists,
                                                            self.constants.percentile, axis=0))
            bandwidth_normalized[bandwidth_normalized < np.percentile(bandwidth_normalized, 20)] = \
                np.percentile(bandwidth_normalized, 20)
            bandwidth_normalized /= np.median(bandwidth_normalized)
            # bandwidth_normalized *= 10
            scores = [self.score_leave_one_out(bandwidth=datapoints[1]*bandwidth_normalized),
                      self.score_leave_one_out(bandwidth=datapoints[2]*bandwidth_normalized)]
        at_boundary_min = False  # Check if we only search at one side as this could indicate ...
        at_boundary_max = False  # ... wrong values of min_bw and max_bw
        for _ in range(n_iter):
            if scores[0] > scores[1]:
                at_boundary_min = True
                datapoints[3] = datapoints[2]
                datapoints[2] = datapoints[1]
                scores[1] = scores[0]
                difference = self.constants.invgr * difference
                datapoints[1] = datapoints[0] + self.constants.invgr2 * difference
                if not self.constants.variable_bandwidth:
                    scores[0] = self.score_leave_one_out(bandwidth=datapoints[1])
                else:
                    scores[0] = self.score_leave_one_out(
                        bandwidth=datapoints[1] * bandwidth_normalized)
            else:
                at_boundary_max = True
                datapoints[0] = datapoints[1]
                datapoints[1] = datapoints[2]
                scores[0] = scores[1]
                difference = self.constants.invgr * difference
                datapoints[2] = datapoints[0] + self.constants.invgr * difference
                if not self.constants.variable_bandwidth:
                    scores[1] = self.score_leave_one_out(bandwidth=datapoints[2])
                else:
                    scores[1] = self.score_leave_one_out(
                        bandwidth=datapoints[2] * bandwidth_normalized)

        if not self.constants.variable_bandwidth:
            self.bandwidth = (datapoints[0] + datapoints[2]) / 2 if scores[0] < scores[1] else \
                (datapoints[3] + datapoints[1]) / 2
        else:
            self.bandwidth = (datapoints[0]+datapoints[2])/2*bandwidth_normalized \
                if scores[0] < scores[1] else (datapoints[3]+datapoints[1])/2*bandwidth_normalized

        # Check if we only searched on one side
        if not at_boundary_min:
            # print("Warning: only searched on right side. Might need to increase max_bw.")
            return 1
        if not at_boundary_max:
            # print("Warning: only searched on left side. Might need to decrease min_bw.")
            return 2
        return 0

    def score_leave_one_out(self, bandwidth: Union[float, np.ndarray] = None,
                            include_const: bool = False) -> float:
        """ Return the leave-one-out score.

        The score is based on the first n datapoints, specified using self.constants.n.

        :param bandwidth: Optional bandwidth to be used when computing the score.
        :param include_const: Whether to add the constant value.
        :return: Leave-one-out score.
        """
        # Check if the distance matrix is defined. If not, create it (this takes some time)
        if not self.data_helpers.mindists.size:
            self.data_helpers.mindists = dist.squareform(dist.pdist(self.data,
                                                                    metric='sqeuclidean')) / 2
            self.data_helpers.mindists *= -1  # Do it this way to prevent invalid warning

        # Compute the one-leave-out score
        bandwidth = self.bandwidth if bandwidth is None else bandwidth
        if not self.constants.variable_bandwidth:
            score = (np.sum(np.log(np.sum(np.exp(self.data_helpers.mindists
                                                 [:self.constants.ndata, :self.constants.ndata] /
                                                 bandwidth ** 2),
                                          axis=0) - 1)) -
                     self.constants.ndata * self.constants.dim * np.log(bandwidth))
        else:
            bandwidth_dim = bandwidth**self.constants.dim
            score = np.sum(np.log(np.sum(np.exp(self.data_helpers.mindists
                                                [:self.constants.ndata, :self.constants.ndata] /
                                                bandwidth**2) / bandwidth_dim, axis=1) -
                                  1/bandwidth_dim))
        if include_const:
            score += self.constants.const_score
        return score

    def set_bandwidth(self, bandwidth: Union[float, np.ndarray]) -> None:
        """ Set the bandwidth of the KDE

        Nothing is done other than setting the bandwidth attribute.

        :param bandwidth: float
        """
        self.bandwidth = bandwidth
        if isinstance(self.bandwidth, float):
            self.constants.variable_bandwidth = False
        elif isinstance(self.bandwidth, np.ndarray):
            self.constants.variable_bandwidth = True
        else:
            raise TypeError("Bandwidth must be of type <float> or <np.ndarray>.")

    def set_score_samples(self, xdata: np.ndarray, compute_difference: bool = False) -> None:
        """ Set the data that is to be used to compute the score samples

        By default, the difference is not computed, because this requires a lot of memory.

        :param xdata: Input data
        :param compute_difference: Whether to compute the difference or not (default)
        :return: None
        """
        # If the input x is a 1D array, it is assumed that each entry corresponds to a datapoint
        # This might result in an error if x is meant to be a single (multi-dimensional) datapoint
        if len(xdata.shape) == 1:
            xdata = xdata[:, np.newaxis]
        self.data_helpers.newshape = xdata.shape[:-1]
        if len(xdata.shape) == 2:
            self.data_helpers.data_score_samples = xdata.copy()
        if not len(xdata.shape) == 2:
            self.data_helpers.data_score_samples = \
                xdata.reshape((np.prod(self.data_helpers.newshape), xdata.shape[-1]))

        if self.scaling:
            self.data_helpers.data_score_samples /= self.data_helpers.std

        # Compute the distance of the datapoints in x to the datapoints of the KDE
        # Let x have M datapoints, then the result is a (self.constants.n-by-M)-matrix
        # Reason to do this now is that this will save computations when the score needs to be
        # computed multiple times (e.g., with different values of self.constants.n)
        self.data_helpers.data_dist = dist.cdist(self.data, self.data_helpers.data_score_samples,
                                                 metric='sqeuclidean')

        # Compute the difference of the datapoints in x to the datapoints of the KDE
        # The different is a n-by-m-by-d matrix, so the vector (i,j,:) corresponds to
        # kde.data[i] - x[j]
        # The difference if only needed to compute the gradient. Therefore, by default, the
        # difference is not computed
        if compute_difference:
            self.data_helpers.difference = \
                np.zeros((len(self.data),
                          len(self.data_helpers.data_score_samples),
                          self.constants.dim))
            for i, xdatam in enumerate(self.data_helpers.data_score_samples):
                self.data_helpers.difference[:, i, :] = self.data - xdatam

    def score_samples(self, xdata: np.ndarray = None) -> np.ndarray:
        """ Return the scores, i.e., the value of the pdf, for all the datapoints in x

        Note that this function will return an error when the bandwidth is not defined. The
        bandwidth can be set using set_bandwidth() or computed using compute_bandwidth().
        If no data is given, it is assumed that the data is already set by set_score_samples()!

        If the input xdata is a 1D array, it is assumed that each entry corresponds to a datapoint.
        This might result in an error if xdata is meant to be a single (multi-dimensional)
        datapoint.

        :param xdata: Input data
        :return: Values of the KDE evaluated at x
        """

        if xdata is None:
            # The data is already set. We can compute the scores directly using _logscore_samples
            scores = np.exp(self._logscore_samples())

            # The data needs to be converted to the original input shape
            return scores.reshape(self.data_helpers.newshape)

        # If the input xdata is a 1D array, it is assumed that each entry corresponds to a
        # datapoint
        # This might result in an error if xdata is meant to be a single (multi-dimensional)
        # datapoint
        if len(xdata.shape) == 1:
            xdata = xdata[:, np.newaxis]
        if len(xdata.shape) == 2:
            return np.exp(self._logscore_samples(xdata))

        # It is assumed that the last dimension corresponds to the dimension of the data
        # (i.e., a single datapoint)
        # Data is transformed to a 2d-array which can be used by self.kde. Afterwards,
        # data is converted to input shape
        newshape = xdata.shape[:-1]
        scores = np.exp(self._logscore_samples(xdata.reshape((np.prod(newshape),
                                                              xdata.shape[-1]))))
        return scores.reshape(newshape)

    def _logscore_samples(self, xdata: np.ndarray = None) -> np.ndarray:
        """ Return the scores, i.e., the value of the pdf, for all the datapoints in x.

        It is assumed that x is in the correct format, i.e., 2D array.
        NOTE: this function returns the LOG of the scores!!!

        The reason to use this function instead of score_samples from sklearn's KernelDensity is
        that this function takes into account the number of datapoints (i.e., self.constants.n).
        Furthermore, for some reason, this function is approximately 10 times as fast as
        sklearn's function!!!

        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.
        """
        # Compute the distance of the datapoints in x to the datapoints of the KDE
        # Let x have M datapoints, then the result is a (self.constants.n-by-M)-matrix
        if xdata is None:
            eucl_dist = self.data_helpers.data_dist[:self.constants.ndata]
        else:
            if self.scaling:
                xdata = xdata / self.data_helpers.std
            eucl_dist = dist.cdist(self.data[:self.constants.ndata], xdata, metric='sqeuclidean')

        # Note that we have f(x,n) = sum [ (2pi)^(-d/2)/(n h^d) * exp{-(x-xi)^2/(2h**2)} ]
        #                          = (2pi)^(-d/2)/(n h^d) * sum_{i=1}^n [ exp{-(x-xi)^2/(2h**2)} ]
        # We first compute the sum. Then the log of f(x,n) is computed:
        # log(f(x,n)) = -d/2*log(2pi) - log(n) - d*log(h) + log(sum)
        sum_kernel = np.zeros(eucl_dist.shape[1])
        if not self.constants.variable_bandwidth:
            for dimension in eucl_dist:
                sum_kernel += np.exp(-dimension / (2 * self.bandwidth**2))
            const = (-self.constants.dim/2*np.log(2*np.pi) - np.log(self.constants.ndata) -
                     self.constants.dim * np.log(self.bandwidth))
        else:
            for dimension, sample_bandwidth in zip(eucl_dist,
                                                   self.bandwidth[:self.constants.ndata]):
                sum_kernel += (np.exp(-dimension / (2 * sample_bandwidth**2)) /
                               sample_bandwidth**self.constants.dim)
            const = -self.constants.dim/2*np.log(2*np.pi) - np.log(self.constants.ndata)

        if self.scaling:
            const -= np.sum(np.log(self.data_helpers.std))

        return const + np.log(sum_kernel)

    def cdf(self, xdata: np.ndarray = None) -> np.ndarray:
        """ Compute the cumulative distribution function of the KDE

        Note that this function will return an error when the bandwidth is not defined. The
        bandwidth can be set using set_bandwidth() or computed using compute_bandwidth().
        If no data is given, it is assumed that the data is already set by set_score_samples()!

        If the input xdata is a 1D array, it is assumed that each entry corresponds to a datapoint.
        This might result in an error if xdata is meant to be a single (multi-dimensional)
        datapoint.

        :param xdata: Input data
        :return: Values of the KDE evaluated at x
        """
        xdata = xdata.copy()
        if xdata is None:
            xdata = self.data_helpers.data_score_samples
        return process_reshaped_data(xdata, self._cdf)

    def _cdf(self, xdata: np.ndarray) -> np.ndarray:
        if self.scaling:
            xdata /= self.data_helpers.std
        cdf = np.ones((self.constants.ndata, len(xdata)))
        data = self.data[:self.constants.ndata] / (np.sqrt(2) * self.bandwidth)
        xdata /= (np.sqrt(2) * self.bandwidth)
        for i in range(self.constants.dim):
            difference = np.subtract(*np.meshgrid(xdata[:, i], data[:, i]))
            cdf *= (scipy.special.erf(difference) + 1) / 2
        cdf = np.mean(cdf, axis=0)
        return cdf

    def gradient_samples(self, xdata: np.ndarray = None) -> np.ndarray:
        """ Compute gradient of the KDE

        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.

        :param xdata: np.ndarray with the datapoints.
        :return: gradient of the KDE
        """
        if xdata is None:
            # The data is already set. We can compute the scores directly using _logscore_samples
            gradient = self._gradient_samples()

            # The data needs to be converted to the original input shape
            return gradient.reshape(self.data_helpers.data_score_samples.shape)

        # If the input x is a 1D array, it is assumed that each entry corresponds to a datapoint
        # This might result in an error if x is meant to be a single (multi-dimensional) datapoint
        if len(xdata.shape) == 1:
            xdata = xdata[:, np.newaxis]
        if len(xdata.shape) == 2:
            return self._gradient_samples(xdata)

        # It is assumed that the last dimension corresponds to the dimension of the data
        # (i.e., a single datapoint)
        # Data is transformed to a 2d-array which can be used by self.kde. Afterwards, data is
        # converted to input shape
        newshape = xdata.shape
        gradient = self._gradient_samples(xdata.reshape((np.prod(newshape[:-1]), xdata.shape[-1])))
        return gradient.reshape(newshape)

    def _gradient_samples(self, xdata: np.ndarray = None) -> np.ndarray:
        """ Compute gradient of the KDE

        It is assumed that the data is already in the right format (i.e., a 2D array). If not, use
        gradient_samples().
        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.

        :param xdata: m-by-d array with m datapoints
        :return: m-by-d vector, where the i-th element corresponds to the gradient at the m-th
            datapoint
        """
        if xdata is None:
            # Assume that we already did the proper calculations
            eucl_dist = self.data_helpers.data_dist[:self.constants.ndata]
            if self.data_helpers.difference is None:
                self.set_score_samples(self.data_helpers.data_score_samples,
                                       compute_difference=True)
            difference = self.data_helpers.difference[:self.constants.ndata]
        else:
            if self.scaling:
                xdata /= self.data_helpers.std
            # Compute the distance of the datapoints in x to the datapoints of the KDE
            # Let x have M datapoints, then the result is a (self.constants.n-by-M)-matrix
            eucl_dist = dist.cdist(self.data[:self.constants.ndata], xdata, metric='sqeuclidean')

            # First compute "difference" = x - xi, which is now a n-by-m-by-d matrix
            difference = np.zeros((self.constants.ndata, len(xdata), self.constants.ndata))
            for i, xdatam in enumerate(xdata):
                difference[:, i, :] = self.data[:self.constants.ndata] - xdatam

        # The gradient is defined as follows:
        # df(x,n)/dx = (2pi)^(-d/2)/(n h^(d+2)) * sum_{i=1}^n [ exp{-(x-xi)^2/(2h**2)} (x - xi) ]
        summation = np.einsum('nm,nmd->md',
                              np.exp(-eucl_dist / (2 * self.bandwidth ** 2)),
                              difference)
        const = (1 / (self.constants.ndata * self.bandwidth ** (self.constants.dim + 2)) /
                 (2 * np.pi) ** (self.constants.dim / 2))
        return const * summation

    def laplacian(self, xdata: np.ndarray = None) -> np.ndarray:
        """ Compute the Laplacian of the KDE

        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.

        :param xdata: m-by-d array with m datapoints
        :return: Laplacian of each of the m datapoints
        """
        if xdata is None:
            # The data is already set. We can compute the scores directly using _logscore_samples
            laplacian = self._laplacian()

            # The data needs to be converted to the original input shape
            return laplacian.reshape(self.data_helpers.newshape)

        # If the input x is a 1D array, it is assumed that each entry corresponds to a datapoint
        # This might result in an error if x is meant to be a single (multi-dimensional) datapoint
        if len(xdata.shape) == 1:
            xdata = xdata[:, np.newaxis]
        if len(xdata.shape) == 2:
            return self._laplacian(xdata)

        # It is assumed that the last dimension corresponds to the dimension of the data
        # (i.e., a single datapoint)
        # Data is transformed to a 2d-array which can be used by self.kde. Afterwards, data is
        # converted to input shape
        newshape = xdata.shape[:-1]
        laplacian = self._laplacian(xdata.reshape((np.prod(newshape), xdata.shape[-1])))
        return laplacian.reshape(newshape)

    def _laplacian(self, xdata: np.ndarray = None) -> np.ndarray:
        """ Compute the Laplacian of the KDE

        It is assumed that the data is already in the right format (i.e., a 2D array). If not, use
        gradient_samples().
        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.

        :param xdata: m-by-d array with m datapoints
        :return: Laplacian of each of the m datapoints
        """
        # Compute the distance of the datapoints in x to the datapoints of the KDE
        # Let x have M datapoints, then the result is a (self.constants.n-by-M)-matrix
        if xdata is None:
            eucl_dist = self.data_helpers.data_dist[:self.constants.ndata]
        else:
            if self.scaling:
                xdata /= self.data_helpers.std
            eucl_dist = dist.cdist(self.data[:self.constants.ndata], xdata, metric='sqeuclidean')

        # The Laplacian is defined as the trace of the Hessian
        # Let one value of a Kernel be denoted by K(u), then the Laplacian for that Kernel is:
        # K(u) * (u^2 - d) / h^2
        # K(u) can be computed as is done in _logscore_samples()  (hereafter, p=K(u))
        # u^2 is the squared euclidean distance divided by h^2, hence, u^2=eucl_dist/kde.bw**2
        # d is the dimension of the data and h is the bandwidth
        laplacian = np.zeros(eucl_dist.shape[1])
        for dimension in eucl_dist:
            pdf = np.exp(-dimension / (2 * self.bandwidth ** 2)) / \
                ((2 * np.pi) ** (self.constants.dim / 2) * self.bandwidth ** self.constants.dim)
            laplacian += pdf * (dimension / self.bandwidth ** 4 - self.constants.dim /
                                self.bandwidth ** 2)
        return laplacian / self.constants.ndata

    def confidence_interval(self, xdata: np.ndarray, confidence: float = 0.95):
        """ Determine the confidence interval

        :param xdata: Input data
        :param confidence: Confidence, by default 0.95
        :return: Upper and lower confidence band
        """
        if len(xdata.shape) == 1:
            xdata = xdata[:, np.newaxis]
        if self.scaling:
            xdata /= self.data_helpers.std
        zvalue = scipy.stats.norm.ppf(confidence/2+0.5)
        density = self.score_samples(xdata)
        std = np.sqrt(self.constants.muk * density / (self.constants.ndata *
                                                      self.bandwidth ** self.constants.dim))
        lower_conf = density - zvalue*std
        upper_conf = density + zvalue*std
        return lower_conf, upper_conf

    def sample(self, n_samples=1):
        """ Generates random samples from the model

        Based on the scikit learn implementation

        Args
        ----
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples.
        """

        uniform_vars = np.random.uniform(0, 1, size=n_samples)
        i = (uniform_vars * self.data.shape[0]).astype(np.int64)
        samples = np.atleast_2d(np.random.normal(self.data[i], self.bandwidth))
        if self.scaling:
            samples *= self.data_helpers.std
        return samples


def process_reshaped_data(xdata: np.ndarray, func: Callable) -> np.ndarray:
    """ Process some data that might be reshaped first.

    :param xdata: The data that serves as the input.
    :param func: The function that is used to process the input.
    :return: The output in the correct shape.
    """
    if len(xdata.shape) == 1:
        xdata = xdata[:, np.newaxis]
    reshape, newshape = False, []
    if len(xdata.shape) > 2:
        reshape = True
        newshape = xdata.shape[:-1]
        xdata = xdata.reshape((np.prod(newshape), xdata.shape[-1]))
    output = func(xdata)
    if reshape:
        return output.reshape(newshape)
    return output


if __name__ == '__main__':
    np.random.seed(0)

    XDATA = np.random.rand(200)
    KERNEL_DENSITY = KDE(data=XDATA)
    KERNEL_DENSITY.compute_bandwidth()

    print(KERNEL_DENSITY.sample(10))
    print("Bandwidth n=200: {:.5f}".format(KERNEL_DENSITY.bandwidth))
    NSTART = 50
    KERNEL_DENSITY = KDE(data=XDATA[:NSTART])
    KERNEL_DENSITY.compute_bandwidth()
    print("Bandwidth n={:d}: {:.5f}".format(NSTART, KERNEL_DENSITY.bandwidth))
    KERNEL_DENSITY.add_data(XDATA[NSTART:])
    KERNEL_DENSITY.compute_bandwidth()
    print("Bandwidth n=200: {:.5f}".format(KERNEL_DENSITY.bandwidth))

    NDATAPOINTS = [100, 500]
    FIGURE, AXS = plt.subplots(1, len(NDATAPOINTS), figsize=(12, 5))

    for ndatapoint, ax in zip(NDATAPOINTS, AXS):
        XDATA = np.random.randn(ndatapoint)
        KERNEL_DENSITY = KDE(data=XDATA)
        t0 = time.time()
        KERNEL_DENSITY.compute_bandwidth()
        t1 = time.time()
        print("Elapsed time: {:.3f} s".format(t1 - t0))
        print("Bandwidth: {:.5f}".format(KERNEL_DENSITY.bandwidth))

        xpdf = np.linspace(-3, 3, 301)
        ypdf = np.exp(-xpdf**2/2) / np.sqrt(2*np.pi)
        ax.plot(xpdf, ypdf, label='Real')
        ax.plot(xpdf, KERNEL_DENSITY.score_samples(xpdf), label='Estimated')
        low, up = KERNEL_DENSITY.confidence_interval(xpdf)
        ax.fill_between(xpdf, low, up, facecolor=[0.5, 0.5, 1], alpha=0.5, label='95% Confidence')

        ax.legend()
        ax.set_title('{:d} datapoints'.format(ndatapoint))
        ax.grid(True)
    plt.show()
