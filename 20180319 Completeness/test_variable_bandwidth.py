""" Test the performance when using a variable bandwidth.

Creation date: 2020 02 26
Author(s): Erwin de Gelder

Modifications:
"""

import matplotlib.pyplot as plt
import numpy as np
from fastkde import KDE
from gaussianmixture import GaussianMixture


def test_kde(nsamples: int, mixture: GaussianMixture, silverman: bool = False,
             variable_bandwidth: bool = False) -> float:
    """ Test the performance of the KDE.

    :param nsamples: Number of train and test samples.
    :param mixture: The Gaussian mixture for generating the samples.
    :param silverman: Whether to use Silverman's rule. If False, cross
        validation is used.
    :param variable_bandwidth: If silverman=False, this determines whether a
        fixed or a variable bandwidth is used.
    :return: Average score on test data.
    """
    # Generate data and fit the KDE.
    samples = mixture.generate_samples(nsamples)
    kde = KDE(samples)
    if silverman:
        kde.set_bandwidth(1.06*np.std(samples)/nsamples**0.2)
    else:
        kde.constants.variable_bandwidth = variable_bandwidth
        kde.constants.percentile = 95
        kde.compute_bandwidth()

    # Test the score based on other data.
    test = mixture.generate_samples(nsamples)
    return np.average(np.log(kde.score_samples(test)))


if __name__ == "__main__":
    # Parameters
    XLIM = [-4., 4.]
    NSAMPLES = 200
    NREPEAT = 100
    np.random.seed(0)

    # Test performance.
    GM = GaussianMixture(np.array([-1, 0, 1]), np.array([0.03, 0.01, 0.02]))
    (XPDF,), YPDF = GM.pdf(minx=[XLIM[0]], maxx=[XLIM[1]], npoints=200)
    RESULT = [test_kde(NSAMPLES, GM) for _ in range(NREPEAT)]
    print("Average result when using cross validation:   {:.2f} +/- {:.2f}"
          .format(np.mean(RESULT), np.std(RESULT)))
    RESULT = [test_kde(NSAMPLES, GM, silverman=True) for _ in range(NREPEAT)]
    print("Average result when using Silverman's rule:   {:.2f} +/- {:.2f}"
          .format(np.mean(RESULT), np.std(RESULT)))
    RESULT = [test_kde(NSAMPLES, GM, variable_bandwidth=True) for _ in range(NREPEAT)]
    print("Average result when using variable bandwidth: {:.2f} +/- {:.2f}"
          .format(np.mean(RESULT), np.std(RESULT)))

    # Show the KDE for one-dimensional data.
    SAMPLES = GM.generate_samples(NSAMPLES)
    MY_KDE = KDE(SAMPLES)
    MY_KDE.compute_bandwidth()
    YKDE = MY_KDE.score_samples(XPDF)
    MY_KDE.constants.variable_bandwidth = True
    MY_KDE.constants.percentile = 50
    MY_KDE.compute_bandwidth()
    YKDE2 = MY_KDE.score_samples(XPDF)

    plt.plot(XPDF, YPDF, 'b-', label="Original")
    plt.plot(XPDF, YKDE, 'g-', label="KDE fixed")
    plt.plot(XPDF, YKDE2, 'r-', label="KDE var")
    plt.xlim(XLIM)
    plt.show()
