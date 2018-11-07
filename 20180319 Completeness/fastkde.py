"""
Kernel Density Estimation

Author
------
Erwin de Gelder

Creation
--------
01 Sep 2018


To do
-----


Modifications
-------------
06 Nov 2018 Improve PEP8 compliancy.

"""

import time
import numpy as np
import scipy.spatial.distance as dist
import scipy.stats
import matplotlib.pyplot as plt


class KDE(object):
    """ Kernel Density Estimation
    """
    def __init__(self, data=None, bw=None):
        self.bandwidth = bw
        self.data = None
        self.constants = {
            'n': 0,                         # Number of datapoints
            'const_score': 0,               # Constant part of leave-one-out score
            'd': 0,                         # Dimension of data
            'muk': 0,                       # integral [ kernel(x)^2 ]
            'last_n': 0,                    # Store n when the bandwidth is computed
            'invgr': (np.sqrt(5) - 1) / 2,  # Inverse of Golden Ratio}
            'invgr2': (3 - np.sqrt(5)) / 2  # 1/gr^2
        }
        self.data_helpers = {
            'mindists': np.array([]),               # Negative (minus) of euclidean distances
            'data_score_samples': np.array([]),     # Scores of each sample of data
            'newshape': np.array([]),               # The new shape of the data to be returned
            'data_dist': np.array([]),              # Euclidean distance of KDE data and input data
            'difference': np.array([]),             # Difference of KDE data and input data
        }
        # self.xhist, self.yhist, self.fft = None, None, None  # Not used at the moment
        self.fit(data)

    def fit(self, data: np.array) -> None:
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

        # Note: creating the distance matrix takes quite some time and is only needed if cross
        # validation is performed.
        # Therefore, this is not done here. Only the first time when cross validation is performed
        self.set_n(len(self.data))
        self.constants['d'] = self.data.shape[1]
        # muk = integral[ kernel(x)^2 ]
        self.constants['muk'] = 1 / (2**self.constants['d'] * np.sqrt(np.pi**self.constants['d']))

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
        self.constants['n'] = ndatapoints
        self.constants['const_score'] = (-ndatapoints * self.constants['d'] / 2 *
                                         np.log(2 * np.pi) - ndatapoints * np.log(ndatapoints - 1))

    def add_data(self, newdata: np.array) -> None:
        """ Add extra data

        The extra data is stored. The number of datapoints is updated. In case the matrix with
        euclidean distances is already computed, this matrix will be updated as well.

        :param newdata: The provided data. When multidimensional data is used, the data should be
            an n-by-d array, with n datapoints and d dimensions.
        """
        if len(newdata.shape) == 1:
            newdata = newdata[:, np.newaxis]
        nnew = len(newdata)

        # Expand the matrix with the distances if this matrix is already defined
        if self.data_helpers['mindists'] is not None:
            newmindists = dist.squareform(dist.pdist(newdata, metric='sqeuclidean')) / 2
            newmindists *= -1  # Do it this way in order to prevent invalid warning
            oldmindists = -dist.cdist(self.data, newdata, metric='sqeuclidean') / 2
            self.data_helpers['mindists'] = \
                np.concatenate((np.concatenate((self.data_helpers['mindists'], oldmindists),
                                               axis=1),
                                np.concatenate((np.transpose(oldmindists), newmindists), axis=1)),
                               axis=0)

        # Update other stuff
        self.data = np.concatenate((self.data, newdata), axis=0)
        self.set_n(self.constants['n'] + nnew)

    def compute_bandwidth(self, **kwargs) -> None:
        """ Compute the bandwidth

        Currently, the Golden Section Search is used for this. See compute_bandwidth_gss for
        more details.

        """
        self.compute_bandwidth_gss(**kwargs)

    def compute_bandwidth_grid(self, min_bw: float = 0.001, max_bw: float = 1.0,
                               n_bw: int = 200) -> None:
        """ Grid search for optimal bandwidth

        :param min_bw: The minimum bandwidth to look for.
        :param max_bw: The maximum bandwidth to look for.
        :param n_bw: The number of bandwidths to look for.
        """
        bandwidths = np.linspace(min_bw, max_bw, n_bw)
        score = np.zeros_like(bandwidths)
        for i, bandwidth in enumerate(bandwidths):
            score[i] = self.score_leave_one_out(bandwidth=bandwidth)
        self.bandwidth = bandwidths[np.argmax(score)]

    def compute_bandwidth_gss(self, min_bw: float = 0.001, max_bw: float = 1., max_iter: int = 100,
                              tol: float = 1e-5) -> None:
        """ Golden section search.

        Given a function f with a single local minimum in
        the interval [a,b], gss returns a subset interval
        [c,d] that contains the minimum with d-c <= tol.

        :param min_bw: The minimum bandwidth to look for.
        :param max_bw: The maximum bandwidth to look for.
        :param max_iter: The maximum number of iterations to perform.
        :param tol: The tolerance that determines when the algorithm is terminated.
        """
        difference = max_bw - min_bw
        datapoints = np.array([min_bw, 0, 0, max_bw])
        datapoints[1] = datapoints[0] + self.constants['invgr2'] * difference
        datapoints[2] = datapoints[0] + self.constants['invgr'] * difference

        # required steps to achieve tolerance
        n_iter = int(np.ceil(np.log(tol / difference) / np.log(self.constants['invgr'])))
        n_iter = max(1, min(n_iter, max_iter))

        scores = [self.score_leave_one_out(bandwidth=datapoints[1]),
                  self.score_leave_one_out(bandwidth=datapoints[2])]
        at_boundary_min = False  # Check if we only search at one side as this could indicate ...
        at_boundary_max = False  # ... wrong values of min_bw and max_bw
        for _ in range(n_iter):
            if scores[0] > scores[1]:
                at_boundary_min = True
                datapoints[3] = datapoints[2]
                datapoints[2] = datapoints[1]
                scores[1] = scores[0]
                difference = self.constants['invgr'] * difference
                datapoints[1] = datapoints[0] + self.constants['invgr2'] * difference
                scores[0] = self.score_leave_one_out(bandwidth=datapoints[1])
            else:
                at_boundary_max = True
                datapoints[0] = datapoints[1]
                datapoints[1] = datapoints[2]
                scores[0] = scores[1]
                difference = self.constants['invgr'] * difference
                datapoints[2] = datapoints[0] + self.constants['invgr'] * difference
                scores[1] = self.score_leave_one_out(bandwidth=datapoints[2])

        # Check if we only searched on one side
        if not at_boundary_min:
            print("Warning: only searched on right side. Might need to increase max_bw.")
        if not at_boundary_max:
            print("Warning: only searched on right side. Might need to increase max_bw.")

        if scores[0] < scores[1]:
            self.bandwidth = (datapoints[0] + datapoints[2]) / 2
        else:
            self.bandwidth = (datapoints[3] + datapoints[1]) / 2

    def score_leave_one_out(self, bandwidth: float = None) -> float:
        """ Return the leave-one-out score.

        The score is based on the first n datapoints, specified using self.constants['n'].

        :param bandwidth: Optional bandwidth to be used when computing the score.
        :return: Leave-one-out score.
        """
        # Check if the distance matrix is defined. If not, create it (this takes some time)
        if not self.data_helpers['mindists'].size:
            self.data_helpers['mindists'] = dist.squareform(dist.pdist(self.data,
                                                                       metric='sqeuclidean')) / 2
            self.data_helpers['mindists'] *= -1  # Do it this way to prevent invalid warning

        # Compute the one-leave-out score
        bandwidth = self.bandwidth if bandwidth is None else bandwidth
        score = (np.sum(np.log(np.sum(np.exp(self.data_helpers['mindists']
                                             [:self.constants['n'], :self.constants['n']] /
                                             bandwidth ** 2),
                                      axis=0) - 1)) -
                 self.constants['n'] * self.constants['d'] * np.log(bandwidth) +
                 self.constants['const_score'])
        return score

    def set_bandwidth(self, bandwidth: float) -> None:
        """ Set the bandwidth of the KDE

        Nothing is done other than setting the bandwidth attribute.

        :param bandwidth: float
        """
        self.bandwidth = bandwidth

    def set_score_samples(self, xdata: np.array, compute_difference: bool = False) -> None:
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
        self.data_helpers['newshape'] = xdata.shape[:-1]
        if len(xdata.shape) == 2:
            self.data_helpers['data_score_samples'] = xdata.copy()
        if not len(xdata.shape) == 2:
            self.data_helpers['data_score_samples'] = \
                xdata.reshape((np.prod(self.data_helpers['newshape']), xdata.shape[-1]))

        # Compute the distance of the datapoints in x to the datapoints of the KDE
        # Let x have M datapoints, then the result is a (self.constants['n']-by-M)-matrix
        # Reason to do this now is that this will save computations when the score needs to be
        # computed multiple times (e.g., with different values of self.constants['n'])
        self.data_helpers['data_dist'] = dist.cdist(self.data,
                                                    self.data_helpers['data_score_samples'],
                                                    metric='sqeuclidean')

        # Compute the difference of the datapoints in x to the datapoints of the KDE
        # The different is a n-by-m-by-d matrix, so the vector (i,j,:) corresponds to
        # kde.data[i] - x[j]
        # The difference if only needed to compute the gradient. Therefore, by default, the
        # difference is not computed
        if compute_difference:
            self.data_helpers['difference'] = \
                np.zeros((len(self.data),
                          len(self.data_helpers['data_score_samples']),
                          self.constants['d']))
            for i, xdatam in enumerate(self.data_helpers['data_score_samples']):
                self.data_helpers['difference'][:, i, :] = self.data - xdatam

    def score_samples(self, xdata: np.array = None) -> np.array:
        """ Return the scores, i.e., the value of the pdf, for all the datapoints in x

        Note that this function will return an error when the bandwidth is not defined. The
        bandwidth can be set using set_bw() or computed using compute_bw().
        If no data is given, it is assumed that the data is already set by set_score_samples()!

        :param xdata: Input data
        :return: Values of the KDE evaluated at x
        """

        if xdata is None:
            # The data is already set. We can compute the scores directly using _logscore_samples
            scores = np.exp(self._logscore_samples())

            # The data needs to be converted to the original input shape
            return scores.reshape(self.data_helpers['newshape'])

        # If the input x is a 1D array, it is assumed that each entry corresponds to a
        # datapoint
        # This might result in an error if x is meant to be a single (multi-dimensional)
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

    def _logscore_samples(self, xdata: np.array = None) -> np.array:
        """ Return the scores, i.e., the value of the pdf, for all the datapoints in x.
        It is assumed that x is in the correct format, i.e., 2D array.
        NOTE: this function returns the LOG of the scores!!!

        The reason to use this function instead of score_samples from sklearn's KernelDensity is
        that this function takes into account the number of datapoints (i.e., self.constants['n']).
        Furthermore, for some reason, this function is approximately 10 times as fast as
        sklearn's function!!!

        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.
        """
        # Compute the distance of the datapoints in x to the datapoints of the KDE
        # Let x have M datapoints, then the result is a (self.constants['n']-by-M)-matrix
        if xdata is None:
            eucl_dist = self.data_helpers['data_dist'][:self.constants['n']]
        else:
            eucl_dist = dist.cdist(self.data[:self.constants['n']], xdata, metric='sqeuclidean')

        # Note that we have f(x,n) = sum [ (2pi)^(-d/2)/(n h^d) * exp{-(x-xi)^2/(2h**2)} ]
        #                          = (2pi)^(-d/2)/(n h^d) * sum_{i=1}^n [ exp{-(x-xi)^2/(2h**2)} ]
        # We first compute the sum. Then the log of f(x,n) is computed:
        # log(f(x,n)) = -d/2*log(2pi) - log(n) - d*log(h) + log(sum)
        sum_kernel = np.zeros(eucl_dist.shape[1])
        for dimension in eucl_dist:
            sum_kernel += np.exp(-dimension / (2 * self.bandwidth ** 2))
        const = -self.constants['d']/2*np.log(2*np.pi) - np.log(self.constants['n']) - \
            self.constants['d']*np.log(self.bandwidth)
        return const + np.log(sum_kernel)

    def gradient_samples(self, xdata: np.array = None) -> np.array:
        """ Compute gradient of the KDE

        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.

        :param xdata: np.array with the datapoints.
        :return: gradient of the KDE
        """
        if xdata is None:
            # The data is already set. We can compute the scores directly using _logscore_samples
            gradient = self._gradient_samples()

            # The data needs to be converted to the original input shape
            return gradient.reshape(self.data_helpers['data_score_samples'].shape)

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

    def _gradient_samples(self, xdata: np.array = None) -> np.array:
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
            eucl_dist = self.data_helpers['data_dist'][:self.constants['n']]
            if self.data_helpers['difference'] is None:
                self.set_score_samples(self.data_helpers['data_score_samples'],
                                       compute_difference=True)
            difference = self.data_helpers['difference'][:self.constants['n']]
        else:
            # Compute the distance of the datapoints in x to the datapoints of the KDE
            # Let x have M datapoints, then the result is a (self.constants['n']-by-M)-matrix
            eucl_dist = dist.cdist(self.data[:self.constants['n']], xdata, metric='sqeuclidean')

            # First compute "difference" = x - xi, which is now a n-by-m-by-d matrix
            difference = np.zeros((self.constants['n'], len(xdata), self.constants['n']))
            for i, xdatam in enumerate(xdata):
                difference[:, i, :] = self.data[:self.constants['n']] - xdatam

        # The gradient is defined as follows:
        # df(x,n)/dx = (2pi)^(-d/2)/(n h^(d+2)) * sum_{i=1}^n [ exp{-(x-xi)^2/(2h**2)} (x - xi) ]
        summation = np.einsum('nm,nmd->md',
                              np.exp(-eucl_dist / (2 * self.bandwidth ** 2)),
                              difference)
        const = 1 / (self.constants['n'] * self.bandwidth ** (self.constants['d'] + 2)) / \
            (2 * np.pi) ** (self.constants['d'] / 2)
        return const * summation

    def laplacian(self, xdata: np.array = None) -> np.array:
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
            return laplacian.reshape(self.data_helpers['newshape'])

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

    def _laplacian(self, xdata: np.array = None) -> np.array:
        """ Compute the Laplacian of the KDE

        It is assumed that the data is already in the right format (i.e., a 2D array). If not, use
        gradient_samples().
        If no data is given, it is assumed that the data is already set by set_score_samples().
        Therefore, the euclidean distance will not be computed.

        :param xdata: m-by-d array with m datapoints
        :return: Laplacian of each of the m datapoints
        """
        # Compute the distance of the datapoints in x to the datapoints of the KDE
        # Let x have M datapoints, then the result is a (self.constants['n']-by-M)-matrix
        if xdata is None:
            eucl_dist = self.data_helpers['data_dist'][:self.constants['n']]
        else:
            eucl_dist = dist.cdist(self.data[:self.constants['n']], xdata, metric='sqeuclidean')

        # The Laplacian is defined as the trace of the Hessian
        # Let one value of a Kernel be denoted by K(u), then the Laplacian for that Kernel is:
        # K(u) * (u^2 - d) / h^2
        # K(u) can be computed as is done in _logscore_samples()  (hereafter, p=K(u))
        # u^2 is the squared euclidean distance divided by h^2, hence, u^2=eucl_dist/kde.bw**2
        # d is the dimension of the data and h is the bandwidth
        laplacian = np.zeros(eucl_dist.shape[1])
        for dimension in eucl_dist:
            pdf = np.exp(-dimension / (2 * self.bandwidth ** 2)) / \
                ((2 * np.pi) ** (self.constants['d'] / 2) * self.bandwidth ** self.constants['d'])
            laplacian += pdf * (dimension / self.bandwidth ** 4 - self.constants['d'] /
                                self.bandwidth ** 2)
        return laplacian / self.constants['n']

    def confidence_interval(self, xdata: np.array, confidence: float = 0.95):
        """ Determine the confidence interval

        :param xdata: Input data
        :param confidence: Confidence, by default 0.95
        :return: Upper and lower confidence band
        """
        if len(xdata.shape) == 1:
            xdata = xdata[:, np.newaxis]
        zvalue = scipy.stats.norm.ppf(confidence/2+0.5)
        density = self.score_samples(xdata)
        std = np.sqrt(self.constants['muk'] * density / (self.constants['n'] *
                                                         self.bandwidth ** self.constants['d']))
        lower_conf = density - zvalue*std
        upper_conf = density + zvalue*std
        return lower_conf, upper_conf


if __name__ == '__main__':
    np.random.seed(0)

    XDATA = np.random.rand(200)
    KERNEL_DENSITY = KDE(data=XDATA)
    KERNEL_DENSITY.compute_bandwidth()
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
