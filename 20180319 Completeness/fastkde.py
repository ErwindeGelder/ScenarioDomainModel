import numpy as np
import scipy.spatial.distance as dist
import scipy.stats
import time
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


class KDE(object):
    def __init__(self, data=None, bw=None):
        self.kde = KernelDensity()
        self.bw = bw
        self.data, self.mindists, self.n, self.const_score, self.d, self.muk = None, None, 0, 0, 0, 0
        self.fit(data)
        self.invgr = (np.sqrt(5) - 1) / 2  # Inverse of Golden Ratio
        self.invgr2 = (3 - np.sqrt(5)) / 2  # 1/gr^2

    def fit(self, data):
        if len(data.shape) == 1:
            self.data = data[:, np.newaxis]
        else:
            self.data = data
        self.mindists = -dist.squareform(dist.pdist(self.data, metric='sqeuclidean')) / 2
        self.n = len(self.data)
        self.const_score = -self.n/2*np.log(2*np.pi) - self.n*np.log(self.n)
        self.d = self.data.shape[1]
        self.muk = 1 / (2**self.d * np.sqrt(np.pi**self.d))

    def compute_bw(self, **kwargs):
        self.compute_bw_gss(**kwargs)

    def compute_bw_grid(self, min_bw=0.001, max_bw=1, n_bw=200):
        bandwidth = np.linspace(min_bw, max_bw, n_bw)
        score = np.zeros_like(bandwidth)
        for i, h in enumerate(bandwidth):
            score[i] = self.score_leave_one_out(bw=h)
        self.bw = bandwidth[np.argmax(score)]

    ''' Golden section search.                                                                                                                                  

        Given a function f with a single local minimum in                                                                                                       
        the interval [a,b], gss returns a subset interval                                                                                                       
        [c,d] that contains the minimum with d-c <= tol. 
    '''
    def compute_bw_gss(self, min_bw=0.001, max_bw=1, max_iter=100, tol=1e-5):
        diff = max_bw - min_bw
        a, b = min_bw, max_bw
        c = a + self.invgr2 * diff
        d = a + self.invgr * diff

        # required steps to achieve tolerance
        n = int(np.ceil(np.log(tol / diff) / np.log(self.invgr)))
        n = min(n, max_iter)

        yc = self.score_leave_one_out(bw=c)
        yd = self.score_leave_one_out(bw=d)
        for k in range(n):
            if yc > yd:
                b = d
                d = c
                yd = yc
                diff = self.invgr * diff
                c = a + self.invgr2 * diff
                yc = self.score_leave_one_out(bw=c)
            else:
                a = c
                c = d
                yc = yd
                diff = self.invgr * diff
                d = a + self.invgr * diff
                yd = self.score_leave_one_out(bw=d)

        if yc < yd:
            self.bw = (a + d) / 2
        else:
            self.bw = (b + c) / 2

    def score_leave_one_out(self, bw=None):
        h = self.bw if bw is None else bw
        return np.sum(np.log(np.sum(np.exp(self.mindists / h ** 2), axis=0) - 1)) - self.n*np.log(h) + self.const_score

    def compute_kde(self, bw=None):
        if bw is None:
            bw = self.bw
        self.kde.set_params(bandwidth=bw)
        self.kde.fit(self.data)

    def score_samples(self, x):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.exp(self.kde.score_samples(x))

    def confidence_interval(self, x, confidence=0.95):
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        zvalue = scipy.stats.norm.ppf(confidence/2+0.5)
        density = self.score_samples(x)
        std = np.sqrt(self.muk * density / (self.n * self.bw**self.d))
        lower_conf = density - zvalue*std
        upper_conf = density + zvalue*std
        return lower_conf, upper_conf


if __name__ == '__main__':
    np.random.seed(0)
    ndatapoints = [100, 500
                   ]
    f, axs = plt.subplots(1, len(ndatapoints), figsize=(12, 5))

    for ndatapoint, ax in zip(ndatapoints, axs):
        xx = np.random.randn(ndatapoint)
        kde = KDE(data=xx)
        t0 = time.time()
        kde.compute_bw()
        t1 = time.time()
        print("Elapsed time: {:.3f} s".format(t1 - t0))
        print("Bandwidth: {:.5f}".format(kde.bw))
        kde.compute_kde()

        xpdf = np.linspace(-3, 3, 301)
        ypdf = np.exp(-xpdf**2/2) / np.sqrt(2*np.pi)
        ax.plot(xpdf, ypdf, label='Real')
        ax.plot(xpdf, kde.score_samples(xpdf), label='Estimated')
        low, up = kde.confidence_interval(xpdf)
        ax.fill_between(xpdf, low, up, facecolor=[0.5, 0.5, 1], alpha=0.5, label='95% Confidence')
        ax.legend()
        ax.set_title('{:d} datapoints'.format(ndatapoint))
        ax.grid('on')
    plt.show()
