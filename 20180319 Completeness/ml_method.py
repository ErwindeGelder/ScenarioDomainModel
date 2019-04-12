import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os


class ErrorStatistics:
    """Functionality for defining bias and covariance"""

    def __init__(self, ne):
        """Initialize the ErrorStatistics

        Parameters
        ----------
        ne : int
            Dimension of which this class is about
        """
        # Set the dimension of the error
        self.ne = ne

        # Initialize attributes
        self.mu_f = []  # Features for bias
        self.n_mu_f = 0  # Number of features for bias
        self.n_par_mu = 0  # Number of parameters for bias
        self.x0_mu = np.array([])  # Initial values of the parameters for bias
        self.x_mu = np.array([])  # Values of the parameters for bias
        self.mask_vec = []  # Vector mask for all parameters
        self.mask_x_mu = []  # Initial mask
        self.sigma_f = ['1']  # Features for covariance
        self.n_sigma_f = 1  # Number of features for covariance
        self.n_par_sigma = ne * ne  # Number of parameters for covariance
        self.x0_sigma = np.array([np.eye(ne)])  # Initial values of the parameters for covariance
        self.x_sigma = np.array([np.eye(ne)])  # Values of the parameters for covariance
        self.mask_x_sigma = np.zeros((1, 2, 2), dtype=bool)  # Initial mask
        self.X_mu = np.array([])  # Features for mu in numpy format
        self.X_sigma = np.array([])  # Features for sigma in numpy format
        self.E = np.array([])  # Error data (for which statistics are described) in numpy format
        self.N = 0  # Number of datapoints
        self.J0 = 0  # Constant term for the cost function
        self.tex_error_keys = []  # List containing the tex formats of the error keys, e.g. e_x

        # Some parameters
        self.tex_width_first_column = 40
        self.tex_width = 8
        self.tex_cvt_keys = {"1": "1",
                             "df['x']": "$x$", "df['dx']": "$x$", "x": "$x$", "dx": "$x$",
                             "df['y']": "$y$", "df['dy']": "$y$", "y": "$y$", "dy": "$y$",
                             "df['vx']": "$v_x$", "vx": "$v_x$",
                             "df['vy']": "$v_y$", "vy": "$v_y$",
                             "df['ax']": "$a_x$", "ax": "$a_x$",
                             "df['ay']": "$a_y$", "ay": "$a_y$",
                             "df['x']**2": "$x^2$", "df['x']*df['x']": "$x^2", "df['dx']**2": "$x^2$",
                             "df['x']*df['vx']": "$xv_x$", "df['vx']*df['x']": "$xv_x$", "df['dx']*df['vx']": "$xv_x$",
                             "df['x']*df['y']": "$xy$", "df['y']*df['x']": "xy", "df['dx']*df['dy']": "$xy$",
                             "df['x']*df['vy']": "$xv_y$", "df['vy']*df['x']": "$xv_y$", "df['dx']*df['vy']": "$xv_y$",
                             "df['vx']**2": "$v_x^2$", "df['vx']*df['vx']": "$v_x^2",
                             "df['vx']*df['y']": "$yv_x$", "df['y']*df['vx']": "$yv_x$", "df['vx']*df['dy']": "$yv_x$",
                             "df['vx']*df['vy']": "$v_xv_y$", "df['vy']*df['vx']": "$v_xv_y$",
                             "df['y']**2": "$y^2$", "df['y']*df['y']": "$y^2", "df['dy']**2": "$y^2$",
                             "df['y']*df['vy']": "$yv_y$", "df['vy']*df['y']": "$yv_y$", "df['dy']*df['vy']": "$yv_y$",
                             "df['vy']**2": "$v_y^2$", "df['vy']*df['vy']": "$v_y^2"}
        self.tex_cvt_error_keys = {"edx": "$e_{\\textup{x}}$", "ex": "$e_{\\textup{x}}$",
                                   "edy": "$e_{\\textup{y}}$", "ey": "$e_{\\textup{y}}$",
                                   "evx": "$e_{\\textup{vx}}$", "evy": "$e_{\\textup{vy}}$",
                                   "eax": "$e_{\\textup{ax}}$", "eay": "$e_{\\textup{ay}}$",
                                   "e": "$e$"}

    def set_mu_features(self, features):
        """Set the features for mu, i.e. the bias

        Parameters
        ----------
        features : list
            Contains the features, e.g. ["1", "df['x']", "df['vx']"]
        """
        self.mu_f = features
        self.n_mu_f = len(self.mu_f)
        self.n_par_mu = self.n_mu_f * self.ne
        self.x0_mu = np.zeros((self.n_mu_f, self.ne))
        self.x_mu = np.zeros_like(self.x0_mu)
        self.mask_x_mu = np.zeros_like(self.x0_mu, dtype=bool)

    def compute_mu(self):
        """Compute the bias (mu) based on the data"""
        if self.n_mu_f == 0:
            return np.zeros((self.N, self.ne))
        else:
            return np.dot(self.X_mu, self.x_mu)

    def set_sigma_features(self, features):
        """Set the features for sigma, i.e. the covariance

        Parameters
        ----------
        features : list
            Contains the features, e.g. ["1", "df['vx']"]
        """
        self.sigma_f = features
        self.n_sigma_f = len(self.sigma_f)
        self.n_par_sigma = self.n_sigma_f * (self.ne * (self.ne + 1)) // 2
        self.x0_sigma = np.zeros((self.n_sigma_f, self.ne, self.ne))
        self.x_sigma = np.zeros_like(self.x0_sigma)
        self.mask_x_sigma = np.zeros_like(self.x0_sigma, dtype=bool)

        # By default, set initial parameters for feature "1" to the identity matrix
        if "1" in self.sigma_f:
            k = self.sigma_f.index("1")
            self.x0_sigma[k] = np.eye(self.ne)
            self.x_sigma[k] = np.eye(self.ne)

    def compute_sigma(self):
        """Compute the covariance (sigma) based on the data"""
        return np.einsum('ni,ijk->njk', self.X_sigma, self.x_sigma)

    def set_data(self, df, e):
        """Set the data and do already some preprocessing

        The features of the bias (mu) and the covariance (sigma) are
        already computed, so this does not need to be done each time
        when mu or sigma has to be evaluated.

        Parameters
        ----------
        df : panda.DataFrame
            Contains the data. The features are based on this data
        e : panda.DataFrame
            Contains the error data for which the likelihood will be computed
        """

        # Determine number of datapoints
        self.N = len(df)

        # Build matrix with the features for mu
        self.X_mu = np.zeros((self.N, self.n_mu_f))
        for i in range(self.n_mu_f):
            self.X_mu[:, i] = eval(self.mu_f[i])

        # Build matrix with the features for sigma
        self.X_sigma = np.zeros((self.N, self.n_sigma_f))
        for i in range(self.n_sigma_f):
            self.X_sigma[:, i] = eval(self.sigma_f[i])

        # Convert error data to numpy format
        self.E = np.array(e)
        if self.ne == 1:
            self.E = self.E.reshape(self.N, 1)

        # Set constant term for cost function
        self.J0 = self.N * self.ne / 2 * np.log(2 * np.pi)

        # Set tex keys
        self.tex_error_keys = [self.tex_cvt_error_keys[key] if key in self.tex_cvt_error_keys else key
                               for key in e.keys()]

    def get_unbiased_e(self, e=None):
        """Remove the bias from the error

        Parameters
        ----------
        e : panda.DataFrame or numpy.array or None
            Input error. If set to None (default), then orignal data
            set by set_data() will be used

        Returns
        -------
        e_unbiased : panda.DataFrame
            Error data from which the bias is removed
        """

        # Compute the bias
        bias = self.compute_mu()

        # Return data with bias substracted
        if e is None:
            return self.E - bias
        else:
            return e - bias

    def score(self, constraint_penalty=None):
        """Compute likelihood of e, given the data

        Parameters
        ----------
        constraint_penalty : float
            Extra penalty for the reciprocal of the determinants

        Returns
        -------
        score : float
            Total score, which is -log(P) where P is the likelihood
        """

        # Define the bias and compute inbiased features
        mu = self.compute_mu()

        # Define the variance
        sigma = self.compute_sigma()

        # Add second term with all the logs to constant first term
        detsigma = np.linalg.det(sigma)
        j = self.J0 + 0.5 * np.sum(np.log(detsigma))

        # Add third term with all the original exponents
        e_unbiased = self.E - mu
        try:
            j += 0.5 * np.einsum('nj,nj',
                                 np.einsum('ni,nij->nj',
                                           e_unbiased,
                                           np.linalg.inv(sigma)),
                                 e_unbiased)
        except np.linalg.linalg.LinAlgError:
            # We have a singular matrix, so we cannot compute the score
            j = np.nan

        if constraint_penalty is not None:
            j += np.sum(1/detsigma) * constraint_penalty

        # Return the score
        return j

    def score_per_sample(self, constraint_penalty=None):
        """Compute the likelihood of e for each sample

        Parameters
        ----------
        constraint_penalty : float
            Extra penalty for the reciprocal of the determinants

        Returns
        -------
        score : np.array
            Score for each sample
        """

        # Define the bias and compute inbiased features
        mu = self.compute_mu()

        # Define the variance
        sigma = self.compute_sigma()

        # Compute the score
        e_unbiased = self.E - mu
        j = np.ones(self.N) * self.ne / 2 * np.log(2 * np.pi)
        detsigma = np.linalg.det(sigma)
        j += 0.5 * np.log(detsigma)
        j += np.einsum('nj,nj->n',
                       np.einsum('ni,nij->nj',
                                 e_unbiased,
                                 np.linalg.inv(sigma)),
                       e_unbiased)

        # Apply penalty for the constraint
        if constraint_penalty is not None:
            j += constraint_penalty/detsigma

        return j

    def sigma_positive_definiteness(self):
        """Return whether the sigma is always positive definite

        Returns
        -------
        pd : bool
            Whether sigma is positive definite for all samples
        """

        # If no penalty is used, check for positive definiteness
        pd = np.all(np.linalg.eigvals(self.compute_sigma()) > 0)
        return pd

    def constraint(self, x):
        """Compute the constraint for optimization

        The constraint is that the determinant of all sigmas
        need to be positive.

        Parameters
        ----------
        x : array
            Array with to-be-optimized parameters

        Returns
        -------
        c : float
            minimal determinant (which has to be positive)
        """

        # Adapt parameters
        self.set_parameters(x)

        # Compute the sigmas and return minimal determinant
        sigma = self.compute_sigma()
        return np.linalg.det(sigma)

    def set_mask(self, mask_x_mu=None, mask_x_sigma=None):
        """Set the mask that will be used for optimizing the parameters

        With the masks, one can define which parameters need to be optimized.
        If set to false, the initial value will be used for that specific
        parameter.

        Parameters
        ----------
        mask_x_mu : np.array(dtype=bool)
            Mask that defines whether parameters of the bias needs to be optimized
            If false for a certain parameter, it will be set to their initial values
            If set to None (default), it will be ignored
        mask_x_sigma : np.array(dtype=bool)
            Similar to mask_x_mu, but for parameters for the covariance
            If set to None (default), it will be ignored
        """

        # Set the masks
        if mask_x_mu is not None:
            self.mask_x_mu = mask_x_mu
        if mask_x_sigma is not None:
            self.mask_x_sigma = mask_x_sigma

        # Compute logical vector with True for a parameter which mask is true
        i = 0
        self.mask_vec = np.zeros(self.n_par_mu + self.n_par_sigma, dtype=bool)
        for mask_row in self.mask_x_mu:
            for mask_entry in mask_row:
                self.mask_vec[i] = mask_entry
                i += 1
        for mask_mat in self.mask_x_sigma:
            for i_row, mask_row in enumerate(mask_mat):
                for i_entry, mask_entry in enumerate(mask_row):
                    if i_row >= i_entry:
                        self.mask_vec[i] = mask_entry
                        i += 1

    def get_full_mask_mu(self, value=True):
        """Return the mask of mu as if all parameters were to be optimized or not

        Parameters
        ----------
        value : bool
            The value of the mast (True if to-be-optimized)

        Returns
        -------
        mask_x_mu : np.array(dtype=bool)
            Mask that defines whether parameters of the bias needs to be optimized
            If false for a certain parameter, it will be set to their initial values
        """
        if value:
            return np.ones_like(self.mask_x_mu)
        else:
            return np.zeros_like(self.mask_x_mu)

    def get_full_mask_sigma(self, value=True):
        """Return the masks of sigma as if all parameters were to be optimized or not

        Parameters
        ----------
        value : bool
            The value of the mast (True if to-be-optimized)

        Returns
        -------
        mask_x_sigma : np.array(dtype=bool)
            Mask that defines whether parameters of the covariance needs to be optimized
            If false for a certain parameter, it will be set to their initial values
        """
        if value:
            mask_sigma = np.zeros_like(self.mask_x_sigma)
            for i in range(mask_sigma.shape[0]):
                for j in range(mask_sigma.shape[1]):
                    mask_sigma[i, j:, j] = True
            return mask_sigma
        else:
            return np.zeros_like(self.mask_x_sigma)

    def get_full_mask(self, value=True):
        """Return the masks if all parameters were to be optimized or not

        Parameters
        ----------
        value : bool
            The value of the mask (True if to-be-optimized)

        Returns
        -------
        mask_x_mu : np.array(dtype=bool)
            Mask that defines whether parameters of the bias needs to be optimized
            If false for a certain parameter, it will be set to their initial values
        mask_x_sigma : np.array(dtype=bool)
            Similar to mask_x_mu, but for parameters for the covariance
        """

        mask_mu = self.get_full_mask_mu(value=value)
        mask_sigma = self.get_full_mask_sigma(value=value)
        return mask_mu, mask_sigma

    def set_full_mask_mu(self, value=True):
        """Set the mask of mu as if all parameters are to be optimized or not

        Parameters
        ----------
        value : bool
            The value of the mast (True if to-be-optimized)
        """
        self.set_mask(mask_x_mu=self.get_full_mask_mu(value=value))

    def set_full_mask_sigma(self, value=True):
        """Set the mask of sigma as if all parameters are to be optimized or not

        Parameters
        ----------
        value : bool
            The value of the mast (True if to-be-optimized)
        """
        self.set_mask(mask_x_sigma=self.get_full_mask_sigma(value=value))

    def set_full_mask(self, value=True):
        """Set the masks as if all parameters are to be optimized or not

        Parameters
        ----------
        value : bool
            The value of the mast (True if to-be-optimized)
        """
        self.set_full_mask_mu(value=value)
        self.set_full_mask_sigma(value=value)

    def turn_on_mask_constants(self, mu=True, sigma=True):
        """Set mask True for constant features

        Parameters
        ----------
        mu : bool
            Whether to do this for the bias parameters (default=True)
        sigma : bool
            Whether to do this for the covariance parameters (default=True)
        """

        # Do this for the bias
        if mu:
            if "1" in self.mu_f:
                i = self.mu_f.index("1")
                self.mask_x_mu[i, :] = np.ones(self.ne, dtype=bool)

        # Do this for the covariance
        if sigma:
            if "1" in self.sigma_f:
                i = self.sigma_f.index("1")
                for j in range(self.ne):
                    self.mask_x_sigma[i, j, :(j + 1)] = np.ones(j + 1, dtype=bool)

    def cleanup_features(self):
        """Removes features that are not used"""
        # First do this for mu
        keep_idx = []
        keep_features = []
        for i_row, mask_row in enumerate(self.mask_x_mu):
            if np.any(mask_row):
                keep_idx.append(i_row)
                keep_features.append(self.mu_f[i_row])
        mask = self.mask_x_mu[keep_idx]
        self.set_mu_features(keep_features)
        self.mask_x_mu = mask.copy()

        # Repeat procedure for sigma
        keep_idx = []
        keep_features = []
        for i_mat, mask_mat in enumerate(self.mask_x_sigma):
            if np.any(mask_mat):
                keep_idx.append(i_mat)
                keep_features.append(self.sigma_f[i_mat])
        mask = self.mask_x_sigma[keep_idx]
        self.set_sigma_features(keep_features)
        self.mask_x_sigma = mask.copy()

    def get_initial_values(self):
        """Construct array with initial values of the parameters

        The size of the array depends on the masks: it equals the total number
        of True values in self.mask_x_mu and self.mask_x_sigma

        Returns
        -------
        x0 : np.array
            Array with the initial values
        """

        # Initialize
        x0 = np.zeros(np.sum(self.mask_x_mu) + np.sum(self.mask_x_sigma))

        # Go through the masks and add parameter if entry is True
        i = 0
        for mask_row, x0_row in zip(self.mask_x_mu, self.x0_mu):
            for mask_entry, x0_entry in zip(mask_row, x0_row):
                if mask_entry:
                    x0[i] = x0_entry
                    i += 1
        for mask_mat, x0_mat in zip(self.mask_x_sigma, self.x0_sigma):
            for mask_row, x0_row in zip(mask_mat, x0_mat):
                for mask_entry, x0_entry in zip(mask_row, x0_row):
                    if mask_entry:
                        x0[i] = x0_entry
                        i += 1

        # Return the result
        return x0

    def get_current_values(self):
        """Construct array with values of the parameters

        The size of the array depends on the masks: it equals the total number
        of True values in self.mask_x_mu and self.mask_x_sigma

        Returns
        -------
        x : np.array
            Array with the initial values
        """

        # Initialize
        x = np.zeros(np.sum(self.mask_x_mu) + np.sum(self.mask_x_sigma))

        # Go through the masks and add parameter if entry is True
        i = 0
        for mask_row, x_row in zip(self.mask_x_mu, self.x_mu):
            for mask_entry, x_entry in zip(mask_row, x_row):
                if mask_entry:
                    x[i] = x_entry
                    i += 1
        for mask_mat, x_mat in zip(self.mask_x_sigma, self.x_sigma):
            for mask_row, x_row in zip(mask_mat, x_mat):
                for mask_entry, x_entry in zip(mask_row, x_row):
                    if mask_entry:
                        x[i] = x_entry
                        i += 1

        # Return the result
        return x

    def get_bounds(self):
        """Compute bounds for the to-be-optimized parameters

        This functions makes use of the masks. Therefore, set the
        masks before calling this function

        Returns
        -------
        bnds : tuple
            ((min0, max0), ..., (minN, maxN))
        """

        # Initialize
        bnds = ()

        # Go through the masks and add parameter if entry is True
        for mask_row in self.mask_x_mu:
            for mask_entry in mask_row:
                if mask_entry:
                    bnds += ((None, None),)
        for i_mat, mask_mat in enumerate(self.mask_x_sigma):
            for i_row, mask_row in enumerate(mask_mat):
                for i_entry, mask_entry in enumerate(mask_row):
                    if mask_entry:
                        if i_row == i_entry and self.sigma_f[i_mat] == '1':
                            bnds += ((1e-3, None),)
                        else:
                            bnds += ((None, None),)

        # Return the result
        return bnds

    def set_parameters(self, x):
        """Set the parameters x_mu and x_sigma using reduced vector

        The input is an array. This needs to be translated to the
        parameter arrays x_mu and x_sigma using the masks.
        No values are returned, the class attributes x_mu and x_sigma
        are directly adjusted.

        Parameters
        ----------
        x : array
            Array with to-be-used parameters. It should have the same size as the number of True values in the masks.
        """

        # Go through the masks and adjust parameter for each True entry
        i = 0
        for i_row, mask_row in enumerate(self.mask_x_mu):
            for i_entry, mask_entry in enumerate(mask_row):
                if mask_entry:
                    self.x_mu[i_row, i_entry] = x[i]
                    i += 1
                else:
                    self.x_mu[i_row, i_entry] = self.x0_mu[i_row, i_entry]
        for i_mat, mask_mat in enumerate(self.mask_x_sigma):
            for i_row, mask_row in enumerate(mask_mat):
                for i_entry, mask_entry in enumerate(mask_row):
                    if i_row >= i_entry:
                        if mask_entry:
                            self.x_sigma[i_mat, i_row, i_entry] = x[i]
                            if not i_row == i_entry:
                                self.x_sigma[i_mat, i_entry, i_row] = x[i]
                            i += 1
                        else:
                            self.x_sigma[i_mat, i_row, i_entry] = self.x0_sigma[i_mat, i_row, i_entry]
                            if not i_row == i_entry:
                                self.x_sigma[i_mat, i_entry, i_row] = self.x0_sigma[i_mat, i_entry, i_row]

    def evaluate_score_for_x(self, x, constraint_penalty=None):
        """Evaluates the score with use of the reduced parameter vector x

        First, the parameters for the bias and the covariance are computed
        using the reduced parameter vector x. Next, the result of the score
        is returned.

        Parameters
        ----------
        x : np.array
            Array with to-be-used parameters. It should have the same size
            as the number of True values in the masks.
        constraint_penalty : float
            Extra penalty for the reciprocal of the determinants

        Returns
        -------
        score : float
            Total score, which is -log(P) where P is the likelihood
        """

        # Adapt parameters
        self.set_parameters(x)

        # Return the score
        return self.score(constraint_penalty=constraint_penalty)

    def first_guess_mu(self):
        """Do a first estimate of mu using linear least squares

        Apparently, this does not work really well. I do not know why this is

        Returns
        -------
        mu : np.array
            The first guess of mu computed using linear least squares
        """

        # First create the big A matrix
        a = np.zeros((self.ne*self.N, self.ne*self.n_mu_f))
        for i in range(self.N):
            for j in range(self.ne):
                a[i*self.ne+j, j*self.n_mu_f:(j+1)*self.n_mu_f] = self.X_mu[i]

        # Compute W*A, where W is the weight matrix
        wa = np.zeros_like(a)
        sigma = self.compute_sigma()
        for i in range(self.N):
            wa[i*self.ne:(i+1)*self.ne] = np.dot(sigma[i], a[i*self.ne:(i+1)*self.ne])

        # Compute the linear least squares solution
        x = np.linalg.solve(np.dot(a.T, wa), np.dot(wa.T, self.E.ravel()))
        return x.reshape([self.ne, self.n_mu_f]).T

    def get_gradient(self, x, constraint_penalty=None):
        """Estimate the gradient using central difference method

        In the first place, it is assumed that the mask is full. Therefore,
        this might be inefficient if the mask is sparse.

        Parameters
        ----------
        x : np.array
            Array with to-be-used parameters. It should have the same size
            as the number of True values in the masks.
        constraint_penalty : float
            Extra penalty for the reciprocal of the determinants

        Returns
        -------
        gradient : np.array
            The gradient is estimated using the central difference method
        """

        # Set the values of x
        self.set_parameters(x)

        # Initialize the gradient
        j = np.zeros(self.n_par_mu + self.n_par_sigma)

        # Determine gradient for mu features
        sigma = self.compute_sigma()
        sigmainv = np.linalg.inv(sigma)
        e_unbiased = self.E - self.compute_mu()
        j[0:self.n_par_mu] = np.einsum('ni,nj->ij',
                                       np.einsum('nij,nj->ni',
                                                 sigmainv,
                                                 -e_unbiased),
                                       self.X_mu).T.ravel()

        # Determine the gradient for Sigma features
        tmp = 0.5 * np.einsum('nij,nf->fij',
                              sigmainv,
                              self.X_sigma)  # First part
        tmp2 = np.einsum('ni,nij->nj',
                         e_unbiased,
                         sigmainv)
        tmp -= 0.5 * np.einsum('nif,nj->fij',
                               np.einsum('ni,nf->nif',
                                         tmp2,
                                         self.X_sigma),
                               tmp2)  # Second part
        if constraint_penalty is not None:
            sigmadet = np.linalg.det(sigma)
            tmp -= constraint_penalty * np.einsum('nij,nf->fij',
                                                  np.einsum('n,nij->nij',
                                                            np.reciprocal(sigmadet),
                                                            sigmainv),
                                                  self.X_sigma)
        idx = self.n_par_mu
        for i_f in range(self.n_sigma_f):
            for idx1 in range(self.ne):
                for idx2 in range(idx1 + 1):
                    if idx1 == idx2:
                        j[idx] = tmp[i_f, idx1, idx2]
                    else:
                        j[idx] = 2 * tmp[i_f, idx1, idx2]
                    idx += 1

        # Return the resulting gradient
        return j[self.mask_vec]

    def optimize(self, method='SLSQP', options=None, verbose=False, save_pickle=None, min_verbose=True,
                 show_end_score=True):
        """Optimize the score with respect to the parameters

        Returns
        -------
        res : result from scipy.minimize or final x when using 'custom' or 'gradient' methods
        """
        if options is None:
            options = {}

        if method == 'custom' or method == 'gradient':
            # Adjust the covariance matrix, such that we always start with a valid solution
            self.set_parameters(self.get_initial_values())
            if '1' in self.sigma_f:
                i_f_constant = self.sigma_f.index('1')
                while np.isnan(self.score()):
                    for i in range(self.ne):
                        self.x_sigma[i_f_constant, i, i] += 0.1
            self.x0_sigma = self.x_sigma

        if method == 'custom':
            # Parameters
            if 'stop' in options.keys():
                stop = options['stop']
            else:
                stop = 1e-3
            if 'penalty' in options.keys():
                penalty = options['penalty']
            else:
                penalty = False
            if 'penalty_constant' in options.keys():
                penalty_constant = options['penalty_constant']
            else:
                penalty_constant = 0.01
            if 'penalty_type_constant' in options.keys():
                penalty_type_constant = options['penalty_type_constant']
            else:
                penalty_type_constant = False

            iteration = 0
            x = self.get_initial_values()
            if show_end_score:
                print("Initial score: {:.3f}".format(self.evaluate_score_for_x(x)))
            step = np.ones_like(x)
            step *= 0.1
            last_line_step = 0.1
            old_gradient = np.zeros_like(x)
            if verbose:
                print("Iteration  Stepsize(#)  Steps      Score  Difference      Step2       Score  Difference")
            while np.max(step) > stop or iteration < 30:
                if penalty:
                    if penalty_type_constant:
                        constraint_penalty = penalty_constant
                    else:
                        constraint_penalty = penalty_constant / (10 + iteration)
                else:
                    constraint_penalty = None
                N = 0  # Number of steps taken
                score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                x_old = x.copy()
                old_score = score
                gradient = np.zeros_like(x)

                # Loop through all parameters
                for i in range(len(x)):
                    x[i] -= step[i]
                    new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                    n = 0
                    if new_score < score:
                        # This is a good direction :)
                        while new_score < score:
                            N += 1
                            n += 1
                            score = new_score
                            gradient[i] -= step[i]
                            if n > 2:
                                step[i] *= 2
                                n = 0
                            x[i] -= step[i]
                            new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                        x[i] += step[i]
                    else:
                        # Try other direction
                        x[i] = x_old[i]
                        x[i] += step[i]
                        new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                        if new_score < score:
                            while new_score < score:
                                N += 1
                                n += 1
                                score = new_score
                                gradient[i] += step[i]
                                if n > 2:
                                    step[i] *= 2
                                    n = 0
                                x[i] += step[i]
                                new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                            x[i] -= step[i]
                        else:
                            x[i] = x_old[i]
                            # Decrease the stepsize of this parameter
                            if step[i] > 1e-15:
                                step[i] /= 2
                    if n > 2:
                        step[i] *= 2
                # Show the result
                iteration += 1
                if verbose:
                    print("{:9d} {:9.2e} {:2d} {:6d} {:10.2f}  {:10.2f}".format(iteration, np.max(step),
                                                                                np.sum(step == np.max(step)),
                                                                                N, score, old_score-score), end="")

                # Do some steps in the direction of the gradient which we computed for free
                # Actually, it is not really gradient (more a 'search direction'), but nevermind
                gradient = 0.1*gradient + 0.9*old_gradient  # Low pass filter the gradient
                old_gradient = gradient.copy()
                scaling = np.sqrt(np.dot(gradient, gradient))
                if scaling > 0:
                    gradient /= scaling
                    old_score = score
                    line_step = last_line_step
                    min_step = last_line_step / 8
                    total_step = 0
                    while line_step >= min_step:
                        xold = x.copy()
                        x += gradient * line_step
                        new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                        if constraint_penalty is None:
                            if not self.sigma_positive_definiteness():
                                new_score = score  # Sigma is not positive definite so don't go into this direction
                        n = 0
                        if new_score < score:
                            while new_score < score:
                                score = new_score
                                total_step += line_step
                                n += 1
                                if n > 2:
                                    n = 0
                                    line_step *= 2
                                    min_step *= 2
                                x += gradient * line_step
                                new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                                # If no penalty is used, check for positive definiteness
                                if constraint_penalty is None:
                                    if not self.sigma_positive_definiteness():
                                        new_score = score  # Sigma is not PD so don't go into this direction
                            x -= gradient * line_step
                        else:
                            x = xold.copy()
                            x -= gradient * line_step
                            new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                            if new_score < score:
                                while new_score < score:
                                    score = new_score
                                    total_step -= line_step
                                    n += 1
                                    if n > 2:
                                        n = 0
                                        line_step *= 2
                                        min_step *= 2
                                    x -= gradient * line_step
                                    new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                                    # If no penalty is used, check for positive definiteness
                                    if constraint_penalty is None:
                                        if not self.sigma_positive_definiteness():
                                            new_score = score  # Sigma is not PD so don't go into this direction
                                x += gradient * line_step
                            else:
                                x = xold.copy()
                        line_step /= 2
                    last_line_step = np.abs(total_step)
                    if last_line_step == 0:
                        last_line_step = np.max([line_step, 1e-15])
                    if verbose:
                        print("  {:9.2e}  {:10.2f}  {:10.2f}".format(total_step, score, old_score-score), end="")
                if verbose:
                    print("")

                # Store intermediate result
                if iteration % 100 == 0:
                    if save_pickle is not None:
                        with open(save_pickle, 'wb') as f:
                            pickle.dump([self.x_mu, self.x_sigma], f)

            self.set_parameters(x)
            if show_end_score:
                print("Final score: {:.3f}".format(self.score()))
            return x
        elif method == 'gradient':
            # Parameters
            if 'tau_end' in options.keys():
                tau_end = options['tau_end']
            else:
                tau_end = 0.01
            if 'tau_start' in options.keys():
                tau_start = options['tau_start']
            else:
                tau_start = 0.2
            if 'tau_N' in options.keys():
                tau_N = options['tau_N']
            else:
                tau_N = 50
            if 'biggest_step' in options.keys():
                biggest_step0 = options['biggest_step']
            else:
                biggest_step0 = 0.1
            if 'N' in options.keys():
                N = options['N']
            else:
                N = 10000
            if 'N2' in options.keys():
                N2 = options['N2']
            else:
                N2 = 10000
            if 'n_check_rate' in options.keys():
                n_check_rate = options['n_check_rate']
            else:
                n_check_rate = 100
            if 'minimum_rate' in options.keys():
                minimum_rate = options['minimum_rate']
            else:
                minimum_rate = 0.01
            if 'penalty' in options.keys():
                penalty = options['penalty']
            else:
                penalty = False

            # Get initial score
            x = self.get_initial_values()
            self.set_parameters(x)
            score = self.score()

            # Increase covariance if needed
            while np.isnan(score):
                # Increase the covariance, such that we have an output which is valid
                if verbose:
                    print("Increase covariance")
                stepsize = 0.1
                for j in range(self.x_sigma.shape[1]):
                    self.x_sigma[0, j, j] += stepsize
                score = self.score()
                self.x0_sigma = self.x_sigma
                x = self.get_initial_values()

            # Do the iterations
            iter = 0
            while True:
                # Initialize stuff
                old_score = self.evaluate_score_for_x(x)
                j = np.sign(self.get_gradient(x))
                j /= np.sqrt(np.dot(j, j))
                if min_verbose:
                    print("Old score: {:.3f}".format(old_score))
                if verbose:
                    print("   #    New score   Difference      Step")
                stop_optimization = False
                scores = np.zeros(N)
                n_very_small_timestep = 0
                biggest_step = biggest_step0

                for q in tqdm(range(N), disable=(not verbose)):
                    iter += 1
                    if penalty:
                        constraint_penalty = 0.1 / (10 + iter) * 100
                    else:
                        constraint_penalty = None
                    score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                    jnew = np.sign(self.get_gradient(x, constraint_penalty=constraint_penalty))
                    # jnew = self.get_gradient(x, constraint_penalty=constraint_penalty)
                    jnew /= np.sqrt(np.dot(jnew, jnew))
                    tau = tau_start - (tau_start - tau_end) * np.minimum(1.0, q / tau_N)
                    j = (1 - tau) * j + tau * jnew
                    j /= np.sqrt(np.dot(j, j))
                    step = biggest_step
                    step_done = False
                    while step >= (biggest_step / 2) or not step_done:
                        vec_step = j * step
                        x -= vec_step
                        new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                        if new_score < score:
                            n_very_small_timestep = 0
                            if not step_done:
                                biggest_step = step
                                step_done = True
                            iteration = 0
                            while new_score < score:
                                score = new_score
                                iteration += 1
                                if iteration >= 3:
                                    break
                                x -= vec_step
                                new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                            if iteration >= 3:
                                step_done = False
                                step *= 2
                                continue
                            x += vec_step
                        else:
                            x += 2 * vec_step
                            new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                            while new_score < score:
                                n_very_small_timestep = 0
                                if not step_done:
                                    biggest_step = step
                                    step_done = True
                                score = new_score
                                x += vec_step
                                new_score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                            x -= vec_step
                        step /= 2
                        if step < 1e-16:
                            n_very_small_timestep += 1
                            if n_very_small_timestep < 3:
                                j = self.get_gradient(x, constraint_penalty=constraint_penalty)
                                j /= np.sqrt(np.dot(j, j)) * 1000
                            else:
                                if verbose:
                                    tqdm.write("Stop optimization because stepsize is very small")
                                stop_optimization = True
                            break
                    if stop_optimization:
                        break
                    if verbose:
                        tqdm.write("{:4d} {:12.3f} {:12.3f}  {:.2e}".format(q, self.evaluate_score_for_x(x),
                                                                            old_score - score, biggest_step))
                        old_score = score
                    scores[q] = score
                    if q > n_check_rate:
                        if (scores[q - n_check_rate] - scores[q]) / n_check_rate < minimum_rate:
                            if verbose:
                                tqdm.write("Stop optimization because costs does not decrease fast enough")
                            break

                # Save progress
                if min_verbose:
                    print("Final score: {:.3f}".format(score))
                self.set_parameters(x)
                with open(save_pickle, 'wb') as f:
                    pickle.dump([self.x_mu, self.x_sigma], f)

                # Now check if we change randomly the parameters, if we can improve
                found_better = False
                n = len(x)
                score = self.evaluate_score_for_x(x, constraint_penalty=constraint_penalty)
                for q in tqdm(range(N2), disable=(not verbose)):
                    new_x = x * (1 + np.random.randn(n) * 2e-3) + np.random.randn(n) * 1e-8
                    new_score = self.evaluate_score_for_x(new_x, constraint_penalty=constraint_penalty)
                    if not np.isnan(new_score):
                        if new_score < score:
                            # Check if eigenvalues are always positive. If not, then skip this one
                            if np.min(np.linalg.eigvals(self.compute_sigma())) <= 0.0:
                                continue
                            x = new_x.copy()
                            found_better = True
                            break
                if not found_better:
                    break
                else:
                    if min_verbose:
                        print("Find better point after {:d} tries".format(q))

            # Return the result
            self.set_parameters(x)
            return x
        else:
            cnstr = {'type': 'ineq',
                     'fun': lambda theta: self.constraint(theta)}
            res = minimize(lambda theta: self.evaluate_score_for_x(theta),
                           self.get_initial_values(), method=method,
                           options=options, jac=self.get_gradient,
                           bounds=self.get_bounds(), constraints=cnstr)
            self.set_parameters(res.x)
            return res

    def pretty_print_x(self, header=True, logfile=None):
        """Present overview of parameters that are used

        Parameters
        ----------
        header : bool
            Whether to print the header
        logfile : str
            If provided, the result is printed to a txt file
        """
        # Width of parameter
        width = 17

        # Check if we need to make a log
        if logfile is not None:
            if not os.path.exists('log'):
                os.mkdir('log')

        # Show the header
        if header:
            my_print("Quantity    Features", logfile)
            my_print("            ", logfile, end="")
            for feature in self.mu_f:
                my_print("{:>{width}s}  ".format(feature, width=width), logfile, end="")
        location_sigma_f = []
        i = len(self.mu_f)
        for j, feature in enumerate(self.sigma_f):
            if feature in self.mu_f:
                location_sigma_f.append(self.mu_f.index(feature))
            else:
                location_sigma_f.append(i)
                i += 1
                if header:
                    my_print("{:>{width}s}  ".format(feature, width=width), logfile, end="")
        if header:
            my_print("", logfile)

        # Show parameters for the bias
        for i in range(self.ne):
            my_print("mu({:d})       ".format(i), logfile, end="")
            for k in range(len(self.mu_f)):
                if self.mask_x_mu[k][i]:
                    my_print("{:{width}.2e}  ".format(self.x_mu[k, i], width=width), logfile, end="")
                else:
                    my_print("{:{width}s}  ".format("", width=width), logfile, end="")
            my_print("", logfile)

        # Show parameters for the covariance
        for i in range(self.ne):
            for j in range(self.ne):
                if np.any(self.mask_x_sigma[:, i, j]):
                    my_print("R({:d},{:d})      ".format(i, j), logfile, end="")
                    for m in range(np.max(location_sigma_f) + 1):
                        if m in location_sigma_f:
                            k = location_sigma_f.index(m)
                            if self.mask_x_sigma[k, i, j]:
                                my_print("{:{width}.2e}  ".format(self.x_sigma[k, i, j], width=width), logfile, end="")
                            else:
                                my_print("{:{width}s}  ".format("", width=width), logfile, end="")
                        else:
                            my_print("{:{width}s}  ".format("", width=width), logfile, end="")
                    my_print("", logfile)

    def export_to_tex(self, filename, e_keys=None, bias=True, cov=True):
        """Export the parameters to a table format in .tex

        Parameters
        ----------
        filename : str
            Filename of the .tex file
        e_keys : List(str)
            List of error keys (this is not used anymore)
        bias : bool
            Whether to print the parameters of the bias
        cov : bool
            Whether to print the parameters of the covariance
        """

        if bias is False and cov is False:
            print("Nothing is printed")
            return

        if e_keys is None:
            e_keys = ['e_x', 'e_{vx}', 'e_y', 'e_{vy}']

        # Compute where the features of sigma should be in the table
        location_sigma_f = []
        if cov and bias:
            i = len(self.mu_f)
            for j, feature in enumerate(self.sigma_f):
                if feature in self.mu_f:
                    location_sigma_f.append(self.mu_f.index(feature))
                else:
                    location_sigma_f.append(i)
                    i += 1
            n_features = np.max([np.max(location_sigma_f)+1, len(self.mu_f)])
        elif cov:
            n_features = len(self.sigma_f)
            location_sigma_f = range(n_features)
        else:
            n_features = len(self.mu_f)

        # Create the file
        with open(filename, 'w') as f:
            # Begin the table
            f.write("\\begin{tabular}{l||")
            for i in range(n_features):
                f.write("r")
            f.write("}\n")

            # Write header
            f.write("    {:{width}s} &".format("Element", width=self.tex_width_first_column))
            f.write(" \\multicolumn{{{:d}}}{{c}}{{Features}} \\\\\n".format(n_features))
            f.write("    {:{width}s}".format("", width=self.tex_width_first_column))
            if bias:
                for feature in self.mu_f:
                    f.write(" & {:{width}s}".format(self.tex_cvt_keys[feature], width=self.tex_width))
            if cov:
                for m in range(len(self.mu_f) if bias else 0, n_features):
                    k = location_sigma_f.index(m)
                    f.write(" & {:{width}s}".format(self.tex_cvt_keys[self.sigma_f[k]], width=self.tex_width))
            f.write(" \\\\ \\hline\\hline\n")

            # Write parameters for the bias
            if bias:
                for i in range(self.ne):
                    f.write("    {:{width}s}".format("Bias({:s})".format(self.tex_error_keys[i]),
                                                     width=self.tex_width_first_column))
                    for k in range(self.n_mu_f):
                        if self.mask_x_mu[k][i]:
                            f.write(" & {:{width}.1e}".format(self.x_mu[k, i], width=self.tex_width))
                        else:
                            f.write(" & {:{width}s}".format("", width=self.tex_width))
                    for k in range(len(self.mu_f), n_features):
                        f.write(" & {:{width}s}".format("", width=self.tex_width))
                    f.write(" \\\\ \\hline\n")

            # Write parameters for the covariance
            if cov:
                for i in range(self.ne):
                    for j in range(self.ne):
                        if np.any(self.mask_x_sigma[:, i, j]):
                            f.write("    {:{width}s}".format("Cov({:s},{:s})".format(self.tex_error_keys[j],
                                                                                     self.tex_error_keys[i]),
                                                             width=self.tex_width_first_column))
                            for m in range(n_features):
                                if m in location_sigma_f:
                                    k = location_sigma_f.index(m)
                                    if self.mask_x_sigma[k, i, j]:
                                        f.write(" & {:{width}.1e}".format(self.x_sigma[k, i, j], width=self.tex_width))
                                    else:
                                        f.write(" & {:{width}s}".format("", width=self.tex_width))
                                else:
                                    f.write(" & {:{width}s}".format("", width=self.tex_width))
                            f.write(" \\\\ \\hline\n")

            # Write end of table
            f.write("\end{tabular}\n")

    def print_constant_bias(self):
        print("Constant bias:")
        for i in range(self.ne):
            print("{:7.3f}".format(np.mean(self.E[:, i])))

    def print_corr_to_tex(self, tex_file=None, unbiased=False, seperater_index=0, corr=None):
        """Print the correlation of the error to a tex tabular

        Parameters
        ----------
        tex_file : str
            If provided, the table will be written to a tex file (otherwise, it will only be printed)
        unbiased : bool
            Whether to use the unbiased error or the original error (default)
        seperater_index : int
            After this number of rows and columns, a row and column seperator will be placed
        corr : np.array
            If set to none (default), correlation will be computed
        """

        # Parameters
        corr_width = 5

        # Compute the correlation
        if corr is None:
            corr = np.corrcoef(self.get_unbiased_e().T if unbiased else self.E.T)

        # Print the correlation
        print("Correlation {:s}biased error:".format("un" if unbiased else ""))
        for i in range(self.ne):
            for j in range(self.ne):
                print(" {:{width}.{decimals}f}".format(corr[i, j], width=corr_width, decimals=(corr_width-3)), end="")
            print("")
        print("")

        # Print correlation to tex file
        if tex_file is not None:
            with open(tex_file, "w") as f:
                # Write header
                f.write("\\begin{tabular}{l|")
                for i in range(self.ne):
                    f.write("r{:s}".format("|" if (i+1) == seperater_index else ""))
                f.write("}\n")
                f.write("    ")
                for i in range(self.ne):
                    f.write(" & {:s}".format(self.tex_error_keys[i]))
                f.write("\\\\ \\hline\n")

                # Write the correlation matrix
                for i in range(self.ne):
                    f.write("    {:s}".format(self.tex_error_keys[i]))
                    for j in range(self.ne):
                        f.write(" & {:{width}.{decimals}f}".format(corr[i, j], width=corr_width,
                                                                   decimals=(corr_width-3)))
                    f.write("\\\\{:s}\n".format(" \\hline" if (i+1) == seperater_index else ""))

                # Write end of table
                f.write("\\end{tabular}\n")

    def print_bootstrap_result(self, x_mu_bootstrap, x_sigma_bootstrap, print_corr=True, logfile=None, tex_file=None,
                               tex_corr_file=None):
        """Print the result from bootstrapping

        Parameters
        ----------
        x_mu_bootstrap : np.array
            Resulting mu features of the bootstraps
        x_sigma_bootstrap : np.array
            Resulting sigma features of the bootstraps
        print_corr : bool
            Whether to print the correlation between parameters
        logfile : str
            If provided, the printed result will also be printed to a text file
        tex_file : str
            If provided, the result will be printed in a tabular in tex format (note that the correlation matrix is
            excluded)
        tex_corr_file : str
            If provided, the resulting correlation matrix will be printed in a tabular tex format (only if
            print_corr=True)

        Returns
        -------
        cov : np.array
            Covariance of the parameters, only if print_corr is true
        """

        # Parameters for printing
        n_par = 10
        n_par_tex = 40
        n_feature = 17
        n_feature_tex = 8
        n_number = 10
        n_number_tex = 9
        n_corr = 5

        # Print the number of bootstraps
        my_print("", logfile)
        my_print("Number of bootstraps: {:d}".format(len(x_mu_bootstrap)), logfile)

        # Print the header
        my_print("", logfile)
        my_print("{:{width}s}  ".format("Parameter", width=n_par), logfile, end="")
        my_print("{:{width}s}  ".format("Feature", width=n_feature), logfile, end="")
        my_print("{:>{width}s}  {:>{width}s}  {:>{width}s}".format("Mean", "Std", "Rel. std", width=n_number), logfile)

        # Print the header to tex
        if tex_file is not None:
            with open(tex_file, "w") as f:
                f.write("\\begin{tabular}{llrrr}\n")
                f.write("    {:{width}s} ".format("Parameter", width=n_par_tex))
                f.write("& {:{width}s} ".format("Feature", width=n_feature_tex))
                f.write("& {:>{width}s} & {:>{width}s} & {:>{width}s}\\\\ \\hline\n".format("Mean", "Std", "Rel.std",
                                                                                            width=n_number_tex))

        # Print the bootstrap results
        n_pars = 0
        for i in range(self.n_par_mu + self.n_par_sigma):
            if self.get_value_mask_index(i):  # Only show result if mask is true
                n_pars += 1  # Count the number of used parameters
                mu, idx = self.get_tuple_index_par(i)
                if mu:
                    mean = np.mean(x_mu_bootstrap[:, idx[1], idx[0]])
                    std = np.std(x_mu_bootstrap[:, idx[1], idx[0]])
                    feature = self.mu_f[idx[1]]
                    my_print("{:{width}s}  ".format("mu({:d})".format(idx[0]), width=n_par), logfile, end="")
                    my_print("{:{width}s}  ".format(feature, width=n_feature), logfile, end="")
                    if tex_file is not None:
                        with open(tex_file, "a") as f:
                            f.write("    {:{width}s} ".format("Bias({:s})".format(self.tex_error_keys[idx[0]]),
                                                              width=n_par_tex))
                            f.write("& {:{width}s} ".format(self.tex_cvt_keys[feature], width=n_feature_tex))
                else:
                    mean = np.mean(x_sigma_bootstrap[:, idx[2], idx[0], idx[1]])
                    std = np.std(x_sigma_bootstrap[:, idx[2], idx[0], idx[1]])
                    feature = self.sigma_f[idx[2]]
                    my_print("{:{width}s}  ".format("sigma({:d},{:d})".format(idx[0], idx[1]), width=n_par), logfile,
                             end="")
                    my_print("{:{width}s}  ".format(feature, width=n_feature), logfile, end="")
                    if tex_file is not None:
                        with open(tex_file, "a") as f:
                            f.write("    {:{width}s} ".format("Cov({:s},{:s})".format(self.tex_error_keys[idx[0]],
                                                                                      self.tex_error_keys[idx[1]]),
                                                              width=n_par_tex))
                            f.write("& {:{width}s} ".format(self.tex_cvt_keys[feature], width=n_feature_tex))
                my_print("{:{width}.1e}  {:{width}.1e}  {:{width}.3f}".format(mean, std, np.abs(std / mean),
                                                                              width=n_number), logfile)
                if tex_file is not None:
                    with open(tex_file, "a") as f:
                        f.write("& {:{width}.1e} & {:{width}.1e} ".format(mean, std, width=n_number_tex))
                        f.write("& {:{width}.3f} \\\\\n".format(np.abs(std / mean), width=n_number_tex))

        # Write end of tex file
        if tex_file is not None:
            with open(tex_file, "a") as f:
                f.write("\\end{tabular}\n")

        if print_corr:
            # Print the correlation matrix for the coefficients
            # First, put all parameters in one big matrix
            pars = np.zeros((len(x_mu_bootstrap), n_pars))
            i_par = 0
            for i in range(self.n_par_mu + self.n_par_sigma):
                if self.get_value_mask_index(i):
                    mu, idx = self.get_tuple_index_par(i)
                    if mu:
                        pars[:, i_par] = x_mu_bootstrap[:, idx[1], idx[0]]
                    else:
                        pars[:, i_par] = x_sigma_bootstrap[:, idx[2], idx[0], idx[1]]
                    i_par += 1

            # Now we can print the correlation
            corr = np.corrcoef(pars.T)
            cov = np.cov(pars.T)
            n_index = int(np.ceil(np.log10(n_pars + 1)))

            # Print the header
            my_print("", logfile)
            my_print("{:{width1}s} {:{width2}s} {:{width3}s}".format("", "", "", width1=n_index, width2=n_par,
                                                                     width3=n_feature), logfile, end="")
            for i in range(n_pars):
                my_print(" {:{width}d}".format(i+1, width=n_corr), logfile, end="")
            my_print("", logfile)

            # Print the header of the texfile
            if tex_corr_file is not None:
                with open(tex_corr_file, 'w') as f:
                    f.write("\\begin{tabular}{ccc|")
                    for _ in range(n_pars):
                        f.write("r")
                    f.write("}\n")
                    f.write("    {:{width1}s} & {:{width2}s} & {:{width3}s}".format("", "", "", width1=n_index,
                                                                                    width2=n_par_tex,
                                                                                    width3=n_feature_tex))
                    for i in range(n_pars):
                        f.write(" & {:{width}d}".format(i+1, width=n_corr))
                    f.write(" \\\\ \\hline\n")

            # Go through the parameters
            i_par = 0
            for i in range(self.n_par_mu + self.n_par_sigma):
                if self.get_value_mask_index(i):
                    my_print("{:{width}d}".format(i_par+1, width=n_index), logfile, end="")
                    if tex_corr_file is not None:
                        with open(tex_corr_file, "a") as f:
                            f.write("    {:{width}d}".format(i_par+1, width=n_index))
                    mu, idx = self.get_tuple_index_par(i)
                    if mu:
                        my_print(" {:{width}s}".format("mu({:d})".format(idx[0]), width=n_par), logfile, end="")
                        my_print(" {:{width}s}".format(self.mu_f[idx[1]], width=n_feature), logfile, end="")
                        if tex_corr_file is not None:
                            with open(tex_corr_file, "a") as f:
                                f.write(" & {:{width}s}".format("Bias({:s})".format(self.tex_error_keys[idx[0]]),
                                                                width=n_par_tex))
                                f.write(" & {:{width}s}".format(self.tex_cvt_keys[self.mu_f[idx[1]]],
                                                                width=n_feature_tex))
                    else:
                        my_print(" {:{width}s}".format("sigma({:d},{:d})".format(idx[0], idx[1]), width=n_par), logfile,
                                 end="")
                        my_print(" {:{width}s}".format(self.sigma_f[idx[2]], width=n_feature), logfile, end="")
                        if tex_corr_file is not None:
                            with open(tex_corr_file, "a") as f:
                                f.write(" & {:{width}s}".format("Cov({:s},{:s})".format(self.tex_error_keys[idx[0]],
                                                                                        self.tex_error_keys[idx[1]]),
                                                                width=n_par_tex))
                                f.write(" & {:{width}s}".format(self.tex_cvt_keys[self.sigma_f[idx[2]]],
                                                                width=n_feature_tex))
                    for j in range(n_pars):
                        my_print(" {:{width}.{decimals}f}".format(corr[i_par, j], width=n_corr, decimals=n_corr-3),
                                 logfile, end="")
                        if tex_corr_file is not None:
                            with open(tex_corr_file, "a") as f:
                                f.write(" & {:{width}.{decimals}f}".format(corr[i_par, j], width=n_corr,
                                                                           decimals=n_corr-3))
                    my_print("", logfile)
                    if tex_corr_file is not None:
                        with open(tex_corr_file, "a") as f:
                            f.write(" \\\\\n")
                    i_par += 1

            # Write end of tabular
            if tex_corr_file is not None:
                with open(tex_corr_file, "a") as f:
                    f.write("\\end{tabular}\n")

            return cov
        else:
            return None

    def sensitivity_plots(self, save_figs=None):
        """Make plots that show the influence of changing parameters around the current x

        Parameters
        ----------
        save_figs : str
            name of figure name, will be appended by the number. Set to none figures do not need to be saved

        Returns
        -------
        f : list
            List of all the plot handles
        """
        # Get the current vector
        x = self.get_current_values()
        nx = len(x)

        # In each plot, we will show the final score
        final_score = self.score()

        # Loop through all values and show the results in different plots
        f = []
        for i in range(nx):
            # Make plot
            f.append(plt.figure(figsize=(3, 2)))
            plt.grid('on')

            # Determine the window size
            xwindow = np.max([2*np.abs(x[i])*1e-4, 1e-6])
            xnew1 = x.copy()
            xnew2 = x.copy()
            xnew1[i] += xwindow/4
            xnew2[i] -= xwindow/4
            while np.isnan(self.evaluate_score_for_x(xnew1)) or np.isnan(self.evaluate_score_for_x(xnew2)):
                xwindow /= 2
                xnew1 = x.copy()
                xnew2 = x.copy()
                xnew1[i] += xwindow/4
                xnew2[i] -= xwindow/4
            xrange = np.linspace(x[i] - xwindow/2, x[i] + xwindow/2, num=25)
            scores = np.zeros_like(xrange)

            # Evaluate the score
            xnew = x.copy()
            for j in range(len(xrange)):
                xnew[i] = xrange[j]
                scores[j] = self.evaluate_score_for_x(xnew)

            # Plot the curve
            plt.plot(xrange, scores, 'b-')
            plt.plot(x[i], final_score, 'k+', markersize=10, markeredgewidth=2)
            plt.xlim((np.min(xrange), np.max(xrange)))
            plt.tight_layout()

            # Save the figure
            if save_figs is not None:
                width = int(np.floor(np.log10(nx)) + 1)
                figname = "{:s}_{:0{width}d}.jpg".format(save_figs, i, width=width)
                plt.savefig(figname)
                plt.close()

        # Set parameters back to original values
        self.set_parameters(x)

        # Return all the plot handles
        return f

    def get_tuple_index_par(self, i):
        """Determine what the i-th parameter is

        Parameters
        ----------
        i : int
            Index of the parameter

        Returns
        -------
        mu : bool
            Whether this is a parameter of mu (True) or sigma (False)
        idx : tuple
            In case of mu=True, this is (ie, if). Otherwise it is
            (ie1, ie2, if). Here ie, ie1 and ie2 indicated the index
            if the error about which the parameter has influence and
            if indicated the index of the feature.
        """

        if i < self.n_par_mu:
            i_f = i // self.ne
            ie = i - i_f * self.ne
            return True, (ie, i_f)
        else:
            i -= self.ne * self.n_mu_f
            i_f = (2 * i) // (self.ne * (self.ne + 1))
            ie = i - i_f * ((self.ne * (self.ne + 1)) // 2)
            ie2 = np.floor((2 * self.ne + 1) / 2 - np.sqrt(((2 * self.ne + 1) / 2) ** 2 - 2 * ie)).astype(np.int)
            ie1 = (ie2 ** 2 / 2 - (2 * self.ne - 1) / 2 * ie2 + ie).astype(np.int)
            return False, (ie1, ie2, i_f)

    def get_scalar_index_par(self, mu, idx):
        """Determine what the (mu,idx)-th parameter is

        Parameters
        ----------
        mu : bool
            Whether this is a parameter of mu (True) or sigma (False)
        idx : tuple
            In case of mu=True, this is (ie, if). Otherwise it is
            (ie1, ie2, if). Here ie, ie1 and ie2 indicated the index
            if the error about which the parameter has influence and
            if indicated the index of the feature.

        Returns
        -------
        i : int
            Index of the parameter
        """

        if mu:
            i = idx[1]*self.ne + idx[0]
        else:
            ie = np.floor((2*self.ne - 1) / 2 * idx[1] - idx[1]**2 / 2).astype(np.int) + idx[0]
            i = idx[2] * (self.ne * (self.ne + 1)) // 2 + ie
            i += self.ne * self.n_mu_f
        return i

    def set_mask_index(self, i, value):
        """Set the mask corresponding to the i-th parameter

        Parameters
        ----------
        i : int
            Index of the parameter
        value : bool
            Value of the mask
        """

        # Retrieve index of the mask
        mu, idx = self.get_tuple_index_par(i)

        # Set mask
        if mu:
            self.mask_x_mu[idx[1]][idx[0]] = value
        else:
            self.mask_x_sigma[idx[2], idx[0], idx[1]] = value

    def get_value_mask_index(self, i):
        """Return the mask value corresponding to the i-th parameter

        Parameters
        ----------
        i : int
            Index of the parameter

        Returns
        -------
        value : bool
            Value of the mask
        """

        # Retrieve index of the mask
        mu, idx = self.get_tuple_index_par(i)

        # Return the value
        if mu:
            return self.mask_x_mu[idx[1]][idx[0]]
        else:
            return self.mask_x_sigma[idx[2], idx[0], idx[1]]

    def print_order_features(self, order_mu, order_sigma, logfile_order, tex_order=None):
        """Print the order of the features in a nice table

        Parameters
        ----------
        order_mu : np.array
            Numpy array of all parameters regarding mu with their order index
        order_sigma : np.array
            Numpy array of all parameters regarding sigma with their order index
        logfile_order : str
            Name of the logfile in which the text will be printen
        tex_order : str
            Name of the tex file in which to print the table
        """

        # Parameters
        width_feature = 20
        width_column = 10

        # Header
        my_print("{:{width}s}".format("Feature", width=width_feature), logfile_order, mode="w", end="")
        for i in range(self.ne):
            my_print(" {:>{width}s}".format("mu({:d})".format(i), width=width_column), logfile_order, end="")
        for j in range(self.ne):
            for i in range(j, self.ne):
                my_print(" {:>{width}s}".format("sigma({:d},{:d})".format(i, j), width=width_column),
                         logfile_order, end="")
        my_print("", logfile_order)
        # Process the features --> it is assumed that mu and sigma have the same features
        for k, f in enumerate(self.mu_f):
            my_print("{:{width}s}".format(f, width=width_feature), logfile_order, end="")
            for i in range(self.ne):
                if order_mu[k, i] == 0:
                    my_print(" {:{width}s}".format("", width=width_column), logfile_order, end="")
                else:
                    if order_mu[k, i] < 1:
                        print(order_mu[k, i])
                    my_print(" {:>{width}.0f}".format(order_mu[k, i], width=width_column), logfile_order, end="")
            if not self.sigma_f[k] == f:
                my_print("Something goes wrong because the features of mu and sigma are not the same", logfile_order)
            for j in range(self.ne):
                for i in range(j, self.ne):
                    l = int(i + (j * (2 * self.ne - 1 - j)) // 2)
                    if order_sigma[k, l] == 0:
                        my_print(" {:{width}s}".format("", width=width_column), logfile_order, end="")
                    else:
                        my_print(" {:>{width}.0f}".format(order_sigma[k, l], width=width_column), logfile_order,
                                 end="")
            my_print("", logfile_order)

        if tex_order is not None:
            with open(tex_order, 'w') as f:
                # Write header
                f.write('\\begin{tabular}{c||')
                for i in range(self.ne + (self.ne * (self.ne + 1)) // 2):
                    if i == self.ne:
                        f.write('|')
                    f.write('c')
                f.write('}\n')
                f.write('    Feature')
                for i in range(self.ne):
                    f.write(' & Bias({:s})'.format(self.tex_error_keys[i]))
                for j in range(self.ne):
                    for i in range(j, self.ne):
                        f.write(' & Cov({:s},{:s})'.format(self.tex_error_keys[i], self.tex_error_keys[j]))
                f.write('\\\\ \\hline \\hline\n')

                # Write the feature order
                for k, feature in enumerate(self.mu_f):
                    if feature == "1":
                        continue
                    f.write('    ' + (self.tex_cvt_keys[feature] if feature in self.tex_cvt_keys.keys() else feature))
                    for i in range(self.ne):
                        f.write(' & ')
                        if order_mu[k, i] > 0:
                            f.write('{:.0f}'.format(order_mu[k, i]))
                    if not self.sigma_f[k] == feature:
                        f.write("Something goes wrong because the features of mu and sigma are not the same")
                    for j in range(self.ne):
                        for i in range(j, self.ne):
                            l = int(i + (j * (2 * self.ne - 1 - j)) // 2)
                            f.write(' & ')
                            if order_sigma[k, l] > 0:
                                f.write('{:.0f}'.format(order_sigma[k, l]))
                    f.write(' \\\\ \\hline\n')

                # Write end of table
                f.write('\\end{tabular}\n')

    def print_result_sfs(self, scores, indices, texfile, imax=100, extra_cols=None, extra_cols_data=None):
        """Apply SFS to retrieve the best features

        Parameters
        ----------
        scores : np.array
            List of all scores
        indices : np.array
            List of indices of parameters in order of there importance
        texfile: str
            The result will be written to this texfile
        imax : int
            Maximum number of features to be printed. By default 100
        extra_cols : list
            Optional list of headings of extra columns
        extra_cols_data : np.array
            Optional, if extra columns, this data will be used
        """
        # Parameters
        w_index = np.ceil(np.log10(len(indices) + 1)).astype(np.int)
        w_element = 40
        w_feature = 7
        w_score = 8

        with open(texfile, 'w') as f:
            # Print the header
            f.write('\\begin{tabular}{rlcr')
            if extra_cols is not None:
                for i in range(len(extra_cols)):
                    f.write('r')
            f.write('}\n')
            f.write('    {:{w1}s} & {:{w2}s} '.format('', 'Element', w1=w_index, w2=w_element))
            f.write('& {:{w1}s} & {:{w2}s} '.format('Feature', 'Score', w1=w_feature, w2=w_score))
            if extra_cols is not None:
                for col in extra_cols:
                    f.write('& {:{w}s} '.format(col, w=w_score))
            f.write('\\\\ \\hline\n')

            # Print the first line
            f.write('    {:>{w1}d} & {:{w2}s} '.format(0, '-', w1=w_index, w2=w_element))
            f.write('& {:{w1}s} & {:>{w2}.0f} '.format('-', scores[0], w1=w_feature, w2=w_score))
            if extra_cols is not None:
                for data in extra_cols_data[0]:
                    f.write('& {:>{w}.0f} '.format(data, w=w_score))
            f.write('\\\\\n')

            # Print the result of the sfs
            for i, (score, index) in enumerate(zip(scores[1:], indices)):
                if i <= imax:
                    mu, idx = self.get_tuple_index_par(int(index))
                    f.write('    {:>{w}d} '.format(i+1, w=w_index))
                    f.write('& {:{w}s} '.format('Bias({:s})'.format(self.tex_error_keys[idx[0]]) if mu else
                                                'Cov({:s},{:s})'.format(self.tex_error_keys[idx[0]],
                                                                        self.tex_error_keys[idx[1]]),
                                                w=w_element))
                    f.write('& {:{w}s} '.format(self.tex_cvt_keys[self.mu_f[idx[1]] if mu else self.sigma_f[idx[2]]],
                                                w=w_feature))
                    f.write('& {:>{w}.0f} '.format(score, w=w_score))
                    if extra_cols is not None:
                        for data in extra_cols_data[i+1]:
                            f.write('& {:>{w}.0f} '.format(data, w=w_score))
                    f.write('\\\\\n')

            # Print end of table
            f.write('\\end{tabular}\n')

    def sequential_forward_selection(self, verbose=True, logfile="log.txt", tex_order=None, only_read_log=False):
        """Apply SFS to retrieve the best features

        Parameters
        ----------
        verbose : bool
            Whether to talk or not
        logfile : str
            Filename of textfile to which the log will be written. Default is 'log.txt'
        tex_order : str
            Filename of the tex file in which to print a table for the order of the parameters
        only_read_log : bool
            If true, then only the result of the logfile will be returned (default=False)

        Returns
        -------
        scores : np.array
            List of all scores
        indices : np.array
            List of indices of parameters in order of there importance
        """
        # Get default values
        bbest_x_mu = self.x0_mu
        bbest_x_sigma = self.x0_sigma

        # By default, turn on all parameters of the feature 1
        npars = self.n_par_mu + self.n_par_sigma
        if "1" in self.mu_f:
            i = self.mu_f.index("1")
            self.mask_x_mu[i, :] = np.ones(self.ne, dtype=bool)
            bbest_x_mu[i, :] = np.mean(self.E, axis=0)
            self.x0_mu = bbest_x_mu.copy()
            npars -= self.ne
        if "1" in self.sigma_f:
            i = self.sigma_f.index("1")
            for j in range(self.ne):
                self.mask_x_sigma[i, j, :(j+1)] = np.ones(j+1, dtype=bool)
            bbest_x_sigma[i, :, :] = np.cov(self.E.T)
            self.x0_sigma = bbest_x_sigma.copy()
            npars -= (self.ne * (self.ne+1)) // 2

        # Store the results
        best_scores = np.zeros(npars + 1)
        best_indices = np.zeros(npars)

        # Look if log file already exists. If so, then use the features which are in there
        has_log = False
        features = []
        n_init_parameters = 0
        order_mu = np.zeros((self.n_mu_f, self.ne))
        order_sigma = np.zeros((self.n_sigma_f, (self.ne * (self.ne + 1)) // 2))
        if os.path.exists(logfile):
            # Read in the log file and use that as starting point
            with open(logfile, "r") as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                if len(content) > 1:
                    has_log = True

                # Process first line for initial score
                line = content[0]
                idx_colon = line.find(':')
                best_scores[0] = float(line[idx_colon + 2:])

                # Process remaining lines
                for iline, line in enumerate(content[1:]):  # Skip first line with initial score
                    # Find score
                    idx_start = line.find('Score:') + 7
                    idx_end = line.find(', Parameter')
                    best_scores[iline+1] = float(line[idx_start:idx_end])

                    # Find parameter
                    idx_parameter = line.find('Parameter')
                    if line[idx_parameter+11:idx_parameter+13] == 'mu':
                        # Find index of mu
                        mu = True
                        idx_end_idx = line.find(')', idx_parameter+13)
                        idx = int(line[idx_parameter+14:idx_end_idx])
                        print("Add mu parameter, idx={:d}".format(idx), end="")
                    else:
                        mu = False
                        idx_end_idx1 = line.find(',', idx_parameter+16)
                        idx_end_idx2 = line.find(')', idx_end_idx1)
                        idx = [int(line[idx_parameter+17:idx_end_idx1]), int(line[idx_end_idx1+1:idx_end_idx2])]
                        print("Add sigma parameter, idx=({:d},{:d})".format(idx[0], idx[1]), end="")
                    idx_feature = line.find('feature')
                    feature = line[idx_feature+9:]
                    print(", feature: {:s}".format(feature))
                    if idx_parameter == -1:
                        print("Cannot process the following line in the log file:")
                        print(line)
                        raise()
                    # Turn on the mask values of the features
                    if mu:
                        if feature in self.mu_f:
                            self.mask_x_mu[self.mu_f.index(feature)][idx] = True
                            features.append([True, idx, self.mu_f.index(feature)])
                            order_mu[self.mu_f.index(feature), idx] = iline + 1
                            best_indices[iline] = self.get_scalar_index_par(True, (idx, self.mu_f.index(feature)))
                        else:
                            print('Cannot find feature "{:s}" in mu features'.format(feature))
                            raise()
                    else:
                        if feature in self.sigma_f:
                            self.mask_x_sigma[self.sigma_f.index(feature)][idx[0]][idx[1]] = True
                            features.append([False, idx[0], idx[1], self.sigma_f.index(feature)])
                            order_sigma[self.sigma_f.index(feature), idx[0]+(idx[1]*(2*self.ne-1-idx[1]))//2] = iline+1
                            best_indices[iline] = self.get_scalar_index_par(False,
                                                                            (idx[0], idx[1], self.mu_f.index(feature)))
                        else:
                            print('Cannot find feature "{:s}" in sigma features'.format(feature))
                            raise()
                    n_init_parameters += 1
            if has_log:
                # Print features list
                print("features = [", end="")
                for i, feature in enumerate(features):
                    if i > 0:
                        print(", ", end="")
                    if i % 5 == 0 and i > 0:
                        print()
                        print("            ", end="")
                    if feature[0]:
                        print("[True, {:d}, {:d}]".format(feature[1], feature[2]), end="")
                    else:
                        print("[False, {:d}, {:d}, {:d}]".format(feature[1], feature[2], feature[3]), end="")
                print("]")

                # Print the order in a nice fashion
                logfile_order = logfile[:-4] + '_feature_order.txt'
                self.print_order_features(order_mu, order_sigma, logfile_order, tex_order)
                if only_read_log:
                    return best_scores, best_indices

                # Do the optimization for these parameters
                self.optimize(method='custom', verbose=False, options={'stop': 1e-4, 'penalty': True},
                              show_end_score=False)
                self.x0_mu = self.x_mu
                self.x0_sigma = self.x_sigma
                x = self.optimize(method='custom', verbose=False, options={'stop': 1e-5, 'penalty': False},
                                  show_end_score=False)
                print("Score with these features: {:.3f}".format(self.evaluate_score_for_x(x)))
                bbest_x_mu = self.x_mu.copy()
                bbest_x_sigma = self.x_sigma.copy()

        # Return this, even if we do not have a result
        if only_read_log:
            return best_scores, best_indices

        # Print initial score
        if not has_log:
            with open(logfile, "w") as f:
                initial_score = self.evaluate_score_for_x(self.get_initial_values())
                f.write("Initial score: {:.3f}\n".format(initial_score))
                print("Initial score: {:.3f}".format(initial_score))
                best_scores[0] = initial_score

        # Do this as many times as there are parameters
        for i in range(npars):
            # Check if we already performed
            if not best_scores[i+1] == 0.0:
                continue

            # Initialize stuff
            best_score = np.inf
            best_index = -1

            # Loop through all parameters
            for j in range(self.n_par_mu + self.n_par_sigma):
                # Check if mask is already True. In that case, skip this one
                if self.get_value_mask_index(j):
                    continue

                if verbose:
                    mu, idx = self.get_tuple_index_par(j)
                    if mu:
                        print("{:45s}".format("Try mu({:d}) feature: {:s}  ".format(idx[0], self.mu_f[idx[1]])),
                              end="")
                    else:
                        print("{:45s}".format("Try sigma({:d},{:d}) feature: {:s}  ".format(idx[0], idx[1],
                                                                                            self.sigma_f[idx[2]])),
                              end="")

                # Set mask at true and do the optimization
                self.x0_mu = bbest_x_mu
                self.x0_sigma = bbest_x_sigma
                self.set_mask_index(j, True)
                self.optimize(method='custom', verbose=False, options={'stop': 1e-4, 'penalty': True},
                              show_end_score=False)
                self.x0_mu = self.x_mu
                self.x0_sigma = self.x_sigma
                x = self.optimize(method='custom', verbose=False, options={'stop': 1e-5, 'penalty': False},
                                  show_end_score=False)
                score = self.evaluate_score_for_x(x)
                if verbose:
                    print("Score: {:.3f}".format(score), end="")
                self.set_mask_index(j, False)

                # Check result and store if it is good
                if score < best_score:
                    best_score = score
                    best_index = j
                    best_x_mu = self.x_mu.copy()
                    best_x_sigma = self.x_sigma.copy()
                    if verbose:
                        print(" --> New best feature up to now")
                else:
                    if verbose:
                        print("")

            # Set best feature 'on' and set initial values accordingly
            if best_index == -1:
                break
            self.set_mask_index(best_index, True)

            # Store the results
            bbest_x_mu = best_x_mu.copy()
            bbest_x_sigma = best_x_sigma.copy()
            best_scores[i+1] = best_score
            best_indices[i] = best_index

            # Print result
            mu, idx = self.get_tuple_index_par(best_index)
            if mu:
                parameter = "mu({:d})".format(idx[0])
                feature = self.mu_f[idx[1]]
            else:
                parameter = "sigma({:d},{:d})".format(idx[0], idx[1])
                feature = self.sigma_f[idx[2]]
            if verbose:
                print("Score: {:.2f}, Parameter: {:s}, feature: {:s}".format(best_score, parameter, feature))
            with open(logfile, "a") as f:
                f.write("{:{width}d} ".format(i + 1, width=int(np.ceil(np.log10(npars+n_init_parameters+1)))))
                f.write("Score: {:.2f}, Parameter: {:s}, feature: {:s}\n".format(best_score, parameter, feature))

        # Return the result
        return best_scores, best_indices


def my_print(string, logfile, mode='a', end='\n'):
    """Write string to file and print it on the screen

    Parameters
    ----------
    string : str
        The string that needs to be printed
    logfile: str
        Name of the logfile. If None, then the writing to the log file will be skipped.
    mode : str
        Mode for opening the logfile. Default is 'a'
    end : str
        End of the string. If None (default), a newline is assumed
    """

    # Write string to logile
    if logfile is not None:
        with open(logfile, mode) as f:
            f.write(string + end)

    # Print text to screen
    print(string, end=end)
