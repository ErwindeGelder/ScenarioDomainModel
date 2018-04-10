"""
Kernel Density Estimation

Author 
------
Jan-Pieter Paardekooper
 
Creation
-------- 
10 May 2016


To do
-----

 
Modifications
-------------
2 January 2017: Make it possible to first scale the data before computing the bandwidth

"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


###############################################################################

class KDE(object):
    """Functionality for Kernel Density Estimation. 


    Provides functionality for computing the KDE and drawing
    random numbers from it.
     
         
    Attributes
    ----------
    data : np.array
        Numpy array with the data
    filename : str
        String with filename of the stored KDE
    bandwidth : float
        Bandwidth used for the KDE
    verbose : bool
        Whether to talk (1) or not (0)
    kde : KernelDensity
        Kernel Density Estimation object for data
    scale_data : bool
        Whether to scale the data before computing the bandwidth such that variance equals one
    std : np.array
        Numpy array with the standard deviation of the data used for scaling the data
     
    To do
    -----
     
     
    Example
    -------

     
    """

    def __init__(self, data=np.empty(0), filename=None, bandwidth=None, verbose=0, scale_data=False):
        """Initialise the class 

        Initialises the data or the filename to load the KDE. One of these 
        needs to be defined in order for the class to work.
         
        Args
        ----
        data : np.array
            Array containing the data
        filename : str
            Filename of the stored KDE
        bandwidth : float
            Bandwidth to use for the KDE
        scale_data : bool
            Whether to scale the data before computing the bandwidth such that variance equals one
        """

        if not data.any() and not filename:
            raise ValueError('Either data or filename needs to be defined')

        self.data = data
        self.filename = filename
        self.bandwidth = bandwidth
        self.verbose = verbose
        self.kde = KernelDensity()
        self.scale_data = scale_data
        self.std = np.empty(0)

    def compute_bandwidth(self, min_bandwidth=0.01, max_bandwidth=0.3, n_bandwidths=30):
        """Compute the optimal bandwidth using cross validation

        Args
        ----
        min_bandwidth : float
            Minimal bandwidth that will be tried
        max_bandwidth : float
            Maximal bandwidth that will be tried
        n_bandwidths : int
            The number of bandwidth choices that will be tried using a grid search
        """

        if self.verbose == 1:
            print('      Computing the optimal bandwidth for the KDE using cross-validation')

        grid = GridSearchCV(self.kde, {'bandwidth': np.linspace(min_bandwidth, max_bandwidth, n_bandwidths)},
                            cv=10)  # 20-fold cross-validation
        if self.scale_data:
            grid.fit(self.data / self.std)
        else:
            grid.fit(self.data)

        if grid.best_params_['bandwidth'] == min_bandwidth:
            if self.verbose == 1:
                print('Warning: the minimum bandwidth is probably too small')

        if grid.best_params_['bandwidth'] == max_bandwidth:
            if self.verbose == 1:
                print('Warning: the maximum bandwidth is probably too small')

        self.bandwidth = grid.best_params_['bandwidth']

        if self.verbose == 1:
            print('      Best bandwidth is {:.3e}'.format(self.bandwidth))
            # print( '      Cross validation score is {:.3e}'.format(grid.best_score_) )

    def compute_kde(self, min_bandwidth=0.01, max_bandwidth=0.3, n_bandwidths=30):
        """Compute the kernel density estimation

        Args
        ----
        min_bandwidth : float
            Minimal bandwidth that will be tried
        max_bandwidth : float
            Maximal bandwidth that will be tried
        n_bandwidths : int
            The number of bandwidth choices that will be tried using a grid search
        """

        # Compute the standard deviation which will be used to scale the data
        if self.scale_data:
            self.std = np.std(self.data, axis=0)

        if self.verbose == 1:
            print('    Computing Kernel Density Estimation')

        if self.bandwidth is None:
            self.compute_bandwidth(min_bandwidth=min_bandwidth, max_bandwidth=max_bandwidth, n_bandwidths=n_bandwidths)

        if self.verbose == 1:
            print('      Computing KDE with optimal bandwidth')

        self.kde = KernelDensity(bandwidth=self.bandwidth)
        if self.scale_data:
            self.kde.fit(self.data / self.std)
        else:
            self.kde.fit(self.data)

    def draw_random_sample(self, n_samples, seed=None):
        """Draw a random sample from the KDE
            
        Args
        ----
        n_samples : int
            Number of random samples to draw
        seed : int
            Seed for random number generator
        """

        if self.verbose == 1:
            print('    Drawing a random sample of %i samples' % n_samples)

        samples = self.kde.sample(n_samples, random_state=seed)
        if self.scale_data:
            samples *= self.std
        return samples

    def lscv(self):
        """Compute the least squares cross validation
        """

        # Number of samples in the data
        nsamples = self.data.shape[0]
        dimsamples = self.data.shape[1]

        # Compute the integral of the squared KDE numerically
        ndim = 100
        min_data = np.min(self.data, axis=0) - 2*self.bandwidth
        max_data = np.max(self.data, axis=0) + 2*self.bandwidth
        points = np.empty((ndim**dimsamples, dimsamples))
        for idim in range(dimsamples):
            column = np.linspace(min_data[idim], max_data[idim], ndim)
            column_rep = [1]
            for i in range(dimsamples):
                if i == idim:
                    column_rep = np.kron(column, column_rep)
                else:
                    column_rep = np.kron(np.ones(ndim), column_rep)
            points[:, idim] = column_rep
        dx = np.product((max_data - min_data) / (ndim - 1))
        intkde2 = np.sum(np.exp(2*self.kde.score_samples(points))) * dx

        scores = np.exp(self.kde.score_samples(self.data))
        offset = 1 / np.sqrt(2 * np.pi) / self.bandwidth / nsamples
        scores_corrected = (scores - offset) / (nsamples - 1) * nsamples
        sumscores = np.sum(scores_corrected) / nsamples * 2

        return intkde2 - sumscores
