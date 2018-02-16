import numpy as np


class DynamicTimeWarping:
    def __init__(self, cost=None, scale_score=False, sakoeband=-1, ikaturaband=-1):
        """
        Initialize DTW object

        :param cost:cost function that takes two scalar arguments and returns the "distance" between the inputs
        :param scale_score: whether to divide score by len(x)+len(y) or not
        """
        self.x = np.array([])
        self.y = np.array([])
        self.cst = np.array([])
        self.acst = np.array([])
        self.nx = 0
        self.ny = 0

        self.scale_score = scale_score
        self.sakoeband = sakoeband
        self.ikaturaband = ikaturaband

        # Define cost function
        if cost is None:
            self.cost = lambda x, y: (x - y)**2
        else:
            self.cost = cost

    def dtw(self, x, y):
        self.x = x
        self.y = y

        self.nx = len(x)
        self.ny = len(y)
        self.cst = np.zeros((self.nx, self.ny))

        # Apply boundary conditions of warping functions
        if self.sakoeband >= 0:
            self.apply_sakoeband()
        if self.ikaturaband > 0:
            self.apply_ikaturaband()

        # Compute cost matrix
        for ix, xx in enumerate(x):
            for iy, yy in enumerate(y):
                if self.cst[ix, iy] == 0.0:
                    self.cst[ix, iy] = self.cost(xx, yy)

        # Compute accumulated cost matrix
        self.acst = self.cst.copy()
        for ix in range(1, self.nx):
            self.acst[ix, 0] += self.acst[ix-1, 0]
        for iy in range(1, self.ny):
            self.acst[0, iy] += self.acst[0, iy-1]
        for ix in range(1, self.nx):
            for iy in range(1, self.ny):
                self.acst[ix, iy] += np.min([self.acst[ix-1, iy], self.acst[ix, iy-1], self.acst[ix-1, iy-1]])

        if self.scale_score:
            scaling = self.nx + self.ny
        else:
            scaling = 1
        return self.acst[-1, -1] / scaling

    def apply_sakoeband(self):
        for ix in range(self.nx):
            for iy in range(self.ny):
                if np.abs(ix - iy) > self.sakoeband:
                    self.cst[ix, iy] = np.inf

    def apply_ikaturaband(self):
        for ix in range(self.nx):
            for iy in range(self.ny):
                s = self.ikaturaband
                if iy < max(ix/s, s*(ix-self.nx+1)+self.ny-1) or iy > min([s*ix, (ix-self.nx+1)/s+self.ny+1]):
                    self.cst[ix, iy] = np.inf

    def compute_warpings(self):
        """
        Compute the warping vector. This is based on the accumalated cost matrix, so it only works after calling dtw()

        :return: wx, wy: warping vectors of signal x and y, respectively
        """

        # Initialize vectors for worst case length
        wx = np.zeros(self.nx+self.ny-1)
        wy = np.zeros_like(wx)

        # We always start at the end!
        wx[0] = self.nx - 1
        wy[0] = self.ny - 1

        # Start iterating from the final end point to the start
        i = 0
        while not (wx[i] == 0 and wy[i] == 0):
            if wx[i] == 0:  # In this case, we are at the boundary of time series x
                wx[i+1] = 0
                wy[i+1] = wy[i] - 1
            elif wy[i] == 0:  # In this case, we are at the boundary of time series y
                wx[i+1] = wx[i] - 1
                wy[i+1] = wy[i]
            else:
                # Pick the three candidates and see which one has the lowest score
                candidates = np.array([self.acst[wx[i]-1, wy[i]],
                                       self.acst[wx[i]-1, wy[i]-1],
                                       self.acst[wx[i], wy[i]-1]])
                argmin = np.argmin(candidates)
                if argmin == 0:
                    wx[i+1], wy[i+1] = wx[i]-1, wy[i]
                elif argmin == 1:
                    wx[i+1], wy[i+1] = wx[i]-1, wy[i]-1
                else:
                    wx[i+1], wy[i+1] = wx[i], wy[i]-1
