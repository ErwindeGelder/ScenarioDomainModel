import numpy as np
from fastkde import fastKDE


class SplineInter1D:
    """Initialize the spline function

    By initializing this, the constraint matrix is built to make future fits faster

    Parameters
    ----------
    t : np.array
        Locations of the knots, including the exterior knots
    d : int
        degree of the spline polynomials
    c : int
        degree of continuity
    left_constraints : list
        Set constraint for left point. For example, [0] means that the zero'th derivative should be 0, [1, 0] means that
        the zero'th derivative should be 1 and the first derivative should be 0. Default is [].
    right_constraints : list
        Idem as left_constraints, but then for the right bound.
    """
    def __init__(self, t, d=3, c=2, left_constraints=None, right_constraints=None):
        # Check options that are given
        if left_constraints is None:
            left_constraints = []
        if right_constraints is None:
            right_constraints = []

        # Store input
        self.t = t
        self.d = d

        # Build constraint matrix
        self.n_coefs = (len(t) - 1) * (d + 1)
        n_interior_constraints = (len(t) - 2) * (c + 1)
        self.n_constraints = n_interior_constraints + len(left_constraints) + len(right_constraints)
        h = np.zeros((self.n_constraints, self.n_coefs))
        g = np.zeros(self.n_constraints)

        # Constuct the matrix with constants, so we have to this only ones
        coefs = np.zeros((c+1, d+1))
        coefs[0, :] = 1
        for i in range(1, c+1):
            for j in range(d-i+1):
                coefs[i, j] = coefs[i-1, j] * (d + 1 - i - j)

        # First do the interior constraints
        for i in range(1, len(t)-1):
            irow = (i - 1) * (c + 1)
            for j in range(c+1):
                for k in range(d-j):
                    coef = coefs[j, k] * t[i]**(d-j-k)
                    h[irow+j, (i-1)*(d+1)+k] = coef
                    h[irow+j, i*(d+1)+k] = -coef
                coef = coefs[j, d-j]
                h[irow+j, (i-1)*(d+1) + d - j] = coef
                h[irow+j, i*(d+1) + d - j] = -coef

        # Add exterior constraints
        n_left_constraints = len(left_constraints)
        for j in range(n_left_constraints):
            for k in range(d-j):
                h[n_interior_constraints+j, k] = coefs[j, k] * t[0]**(d-j-k)
            h[n_interior_constraints+j, d-j] = coefs[j, d-j]
            g[n_interior_constraints+j] = left_constraints[j]
        for j in range(len(right_constraints)):
            for k in range(d-j):
                h[n_interior_constraints+n_left_constraints+j, k-d-1] = coefs[j, k] * t[-1]**(d-j-k)
            h[n_interior_constraints+n_left_constraints+j, -j-1] = coefs[j, d-j]
            g[n_interior_constraints+n_left_constraints+j] = right_constraints[j]

        # Compute using the SVD the thetafixed (the fixed part of the solution)
        u, s, v = np.linalg.svd(h)
        thetabar = np.dot(np.dot(np.diag(np.reciprocal(s)), u.T), g)
        self.thetafixed = np.dot(v[:self.n_constraints, :].T, thetabar)
        self.v2 = v[self.n_constraints:, :].T

    def fit(self, x, y):
        # Construct the vector with indices of the specific interval
        ind = np.searchsorted(self.t, x)
        ind[ind > 0] -= 1

        # Construct the matrix with all monomials
        xbar = np.ones((len(x), self.d+1))
        for k in range(self.d):
            xbar[:, k] = x ** (self.d - k)

        # Construct the big X matrix with all the data (sparse matrix)
        xx = np.zeros((len(x), self.n_coefs))
        for i in range(len(x)):
            xx[i, (self.d + 1) * ind[i] + np.arange(self.d + 1)] = xbar[i, :]

        # Compute the coefficients
        thetatilde, _, _, _ = np.linalg.lstsq(np.dot(xx, self.v2), y-np.dot(xx, self.thetafixed))
        theta = self.thetafixed + np.dot(self.v2, thetatilde)

        # Return spline function
        return _Spline(self.t, self.d, theta)


class _Spline:
    def __init__(self, t, d, theta):
        self.t = t
        self.d = d
        self.theta = theta
        self.x = np.array([])
        self.ind = np.array([])
        self.xbar = np.array([[]])

    def __call__(self, x=None):
        # Construct the vector with indices of the specific interval and the matrix with all monomials
        if x is not None:
            self.x = x
            self.ind_and_xbar()

        # Compute output
        y = np.array([np.dot(self.xbar[i, :],
                             self.theta[self.ind[i]*(self.d+1):(self.ind[i]+1)*(self.d+1)])
                      for i in range(len(self.x))])
        return y

    def ind_and_xbar(self):
        # Construct the vector with indices of the specific interval
        self.ind = np.searchsorted(self.t, self.x)
        self.ind[self.ind > 0] -= 1

        # Construct the matrix with all monomials
        self.xbar = np.ones((len(self.x), self.d + 1))
        for k in range(self.d):
            self.xbar[:, k] = self.x ** (self.d - k)

    def derivative(self, x):
        # Return the derivative...
        # Construct the vector with indices of the specific interval
        ind = np.searchsorted(self.t, x)
        ind[ind > 0] -= 1

        # Construct the matrix with all monomials
        xbar = np.ones((len(x), self.d))
        for k in range(self.d-1):
            xbar[:, k] = (self.d - k) * x ** (self.d - k - 1)

        # Compute output
        y = np.array([np.dot(xbar[i, :], self.theta[ind[i]*(self.d+1):(ind[i]+1)*(self.d+1)-1]) for i in range(len(x))])
        return y

    def dderivative(self, x):
        # Return the double derivative
        # Construct the vector with indices of the specific interval
        ind = np.searchsorted(self.t, x)
        ind[ind > 0] -= 1

        # Construct the matrix with all monomials
        xbar = np.ones((len(x), self.d-1))
        for k in range(self.d-2):
            xbar[:, k] = (self.d - k) * (self.d - k - 1) * x ** (self.d - k - 2)
        xbar[:, self.d-2] = 2

        # Compute output
        y = np.array([np.dot(xbar[i, :], self.theta[ind[i]*(self.d+1):(ind[i]+1)*(self.d+1)-2]) for i in range(len(x))])
        return y

