import numpy as np
import matplotlib.pyplot as plt


# Class for generating data with a mixture of Gaussians (GM)
class GaussianMixture:
    def __init__(self, mu, sigma, weights=None):
        """ Initialise the class

        :param mu: containing the means of the individual Gaussians
        :param sigma: containing the variances of the individual Gaussians
        :param weights: optional weight for the Gaussians. By defaults, the gaussians are equally weighted
        """

        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        if weights is None:
            self.weights = np.ones(len(self.mu)) / len(self.mu)
        else:
            self.weights = np.array(weights) / np.sum(weights)
        self.cweights = np.cumsum(self.weights)

        # Determine the dimension of the GM
        if len(self.mu.shape) == 1:
            self.d = 1
        else:
            self.d = len(self.mu[0])

        # Check if mu and sigma has right shape
        if len(self.mu.shape) == 1:
            self.mu = self.mu[:, np.newaxis]
        if len(self.sigma.shape) == 1:
            self.sigma = self.sigma[:, np.newaxis, np.newaxis]

        # Number of Gaussians
        self.k = self.mu.shape[0]

        # Perform Cholesky factorization that is needed for generation the Gaussians
        self.chol = np.linalg.cholesky(self.sigma)

    def generate_samples(self, n):
        """ Generate samples from the mixture of Gaussians

        :param n: Number of samples
        :return y: n samples from the mixture of Gaussians
        """

        # Determine the index of the gaussian that is to be used
        idx = np.searchsorted(self.cweights, np.random.rand(n))

        # Produce the samples
        x = np.random.randn(n, self.d)
        return self.mu[idx] + np.einsum('nij,nj->ni', self.chol[idx], x)

    def pdf(self, n=51, minx=None, maxx=None):
        """ Generate probability density function (PDF)

        :param n: Number of points to be used for each dimension (default=101)
        :param minx: Lower limit of grid
        :param maxx: Upper limit of grid
        :return x: x values of the pdf
        :return y: height of the pdf
        """

        # Set boundaries of grid
        if minx is None:
            minx = np.zeros(self.d)
            for i in range(self.d):
                minx[i] = np.min(self.mu[:, i] - 3*np.sqrt(self.sigma[:, i, i]))
        if maxx is None:
            maxx = np.zeros_like(minx)
            for i in range(self.d):
                maxx[i] = np.max(self.mu[:, i] + 3*np.sqrt(self.sigma[:, i, i]))

        # Create the PDF grid
        xorig = np.meshgrid(*[np.linspace(minx[i], maxx[i], n) for i in range(self.d)])
        x = np.reshape(xorig, (self.d, n**self.d))

        # Compute determinants and inverses of covariances (needed for computation of PDF)
        det = np.linalg.det(self.sigma)
        invsigma = np.linalg.inv(self.sigma)

        # Evaluate the PDF
        xk = np.kron(x[np.newaxis, :, :], np.ones((self.k, 1, 1)))
        mun = np.kron(self.mu[:, :, np.newaxis], np.ones((1, 1, n**self.d)))
        exp = np.exp(-0.5 * np.einsum('kjn,kjn->kn', np.einsum('kin,kij->kjn', xk - mun, invsigma), xk - mun))
        normalization = 1 / np.sqrt((2*np.pi)**self.d * det)
        pdf = np.einsum('kn,k->n', np.einsum('k,kn->kn', normalization, exp), self.weights)
        return xorig, np.reshape(pdf, tuple([n for _ in range(self.d)]))


if __name__ == "__main__":
    np.random.seed(0)
    gm = GaussianMixture([-1, 1], [0.5, 0.5])
    y = gm.generate_samples(10000)
    plt.hist(y, 100, density=True)
    (xpdf,), ypdf = gm.pdf()
    plt.plot(xpdf, ypdf)
    plt.xlim([xpdf[0], xpdf[-1]])
    plt.title("Histogram of data with plot of real PDF")
    plt.show()

    gm = GaussianMixture([[-1, -1], [1, 1]], [[[1, -.5], [-.5, 1]], [[0.6, -.5], [-.5, 1.4]]])
    y = gm.generate_samples(10000)
    plt.hist2d(y[:, 0], y[:, 1], bins=50, cmap=plt.cm.get_cmap('BuGn_r'))
    (x1pdf, x2pdf), ypdf = gm.pdf()
    plt.contour(x1pdf, x2pdf, ypdf)
    plt.xlim([x1pdf[0, 0], x1pdf[-1, -1]])
    plt.ylim([x2pdf[0, 0], x2pdf[-1, -1]])
    plt.title("Histogram of data with contour plot of real PDF")
    plt.show()
