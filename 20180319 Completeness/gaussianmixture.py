"""
Guassian Mixture

Author
------
Erwin de Gelder

Creation
--------
08 Aug 2018


To do
-----


Modifications
-------------
07 Nov 2018 Make PEP8 compliant.

"""

import numpy as np
import matplotlib.pyplot as plt


# Class for generating data with a mixture of Gaussians (GM)
class GaussianMixture:
    """ Gaussian mixture

    Attributes:
        mean(np.ndarray): Means of the various Gaussians.
        sigma(np.ndarray): Covariances of the various Gaussians.
        weights(np.ndarray): Weights of the various Gaussians.
        cweights(np.ndarray): Cumulative weights of the various Gaussians.
        dim(int): Dimension of the Gaussian Mixture.
        k(int): Number of Gaussians.
        chol(np.ndarray): Cholesky factorization that is needed for generation the Gaussians.
    """
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, weights: np.ndarray = None):
        self.mean = np.array(mu)
        self.sigma = np.array(sigma)
        if weights is None:
            self.weights = np.ones(len(self.mean)) / len(self.mean)
        else:
            self.weights = np.array(weights) / np.sum(weights)
        self.cweights = np.cumsum(self.weights)

        # Determine the dimension of the GM
        if len(self.mean.shape) == 1:
            self.dim = 1
        else:
            self.dim = len(self.mean[0])

        # Check if mu and sigma has right shape
        if len(self.mean.shape) == 1:
            self.mean = self.mean[:, np.newaxis]
        if len(self.sigma.shape) == 1:
            self.sigma = self.sigma[:, np.newaxis, np.newaxis]

        # Number of Gaussians
        self.k = self.mean.shape[0]

        # Perform Cholesky factorization that is needed for generation the Gaussians
        self.chol = np.linalg.cholesky(self.sigma)

    def generate_samples(self, nsamples: int) -> np.ndarray:
        """ Generate samples from the mixture of Gaussians

        :param nsamples: Number of samples
        :return y: n samples from the mixture of Gaussians
        """

        # Determine the index of the gaussian that is to be used
        idx = np.searchsorted(self.cweights, np.random.rand(nsamples))

        # Produce the samples
        xsamples = np.random.randn(nsamples, self.dim)
        return self.mean[idx] + np.einsum('nij,nj->ni', self.chol[idx], xsamples)

    def pdf(self, npoints: int = 51, minx: float = None, maxx: float = None):
        """ Generate probability density function (PDF)

        :param npoints: Number of points to be used for each dimension (default=101).
        :param minx: Lower limit of grid.
        :param maxx: Upper limit of grid.
        :return x: x values of the pdf.
        :return y: height of the pdf.
        """

        # Set boundaries of grid
        if minx is None:
            minx = np.zeros(self.dim)
            for i in range(self.dim):
                minx[i] = np.min(self.mean[:, i] - 3 * np.sqrt(self.sigma[:, i, i]))
        if maxx is None:
            maxx = np.zeros_like(minx)
            for i in range(self.dim):
                maxx[i] = np.max(self.mean[:, i] + 3 * np.sqrt(self.sigma[:, i, i]))

        # Create the PDF grid
        xorig = np.meshgrid(*[np.linspace(minx[i], maxx[i], npoints)
                              for i in range(self.dim)])  # type: Tuple
        marginal_grid = np.reshape(xorig, (self.dim, npoints ** self.dim))

        # Compute determinants and inverses of covariances (needed for computation of PDF)
        det = np.linalg.det(self.sigma)
        invsigma = np.linalg.inv(self.sigma)

        # Evaluate the PDF
        grid = np.kron(marginal_grid[np.newaxis, :, :], np.ones((self.k, 1, 1)))
        mun = np.kron(self.mean[:, :, np.newaxis], np.ones((1, 1, npoints ** self.dim)))
        exp = np.exp(-0.5 * np.einsum('kjn,kjn->kn',
                                      np.einsum('kin,kij->kjn', grid - mun, invsigma),
                                      grid - mun))
        normalization = 1 / np.sqrt((2*np.pi) ** self.dim * det)
        pdf = np.einsum('kn,k->n', np.einsum('k,kn->kn', normalization, exp), self.weights)
        return xorig, np.reshape(pdf, tuple([npoints for _ in range(self.dim)]))


if __name__ == "__main__":
    np.random.seed(0)
    GM = GaussianMixture(np.array([-1, 1]), np.array([0.5, 0.5]))
    YDATA = GM.generate_samples(10000)
    plt.hist(YDATA, 100, density=True)
    (XPDF,), YPDF = GM.pdf()
    plt.plot(XPDF, YPDF)
    plt.xlim([XPDF[0], XPDF[-1]])
    plt.title("Histogram of data with plot of real PDF")
    plt.show()

    GM = GaussianMixture(np.array([[-1, -1], [1, 1]]),
                         np.array([[[1, -.5], [-.5, 1]], [[0.6, -.5], [-.5, 1.4]]]))
    YDATA = GM.generate_samples(10000)
    plt.hist2d(YDATA[:, 0], YDATA[:, 1], bins=50, cmap=plt.cm.get_cmap('BuGn_r'))
    (X1PDF, X2PDF), YPDF = GM.pdf()
    plt.contour(X1PDF, X2PDF, YPDF)
    plt.xlim([X1PDF[0, 0], X1PDF[-1, -1]])
    plt.ylim([X2PDF[0, 0], X2PDF[-1, -1]])
    plt.title("Histogram of data with contour plot of real PDF")
    plt.show()
