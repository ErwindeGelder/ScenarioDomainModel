""" Risk-Taking Behavior Model (RTBM) from Hamdar et al. (2015)

Creation date: 2020 09 23
Author(s): Erwin de Gelder

Modifications:
"""

import numpy as np
from scipy.stats import norm
from .standard_model import StandardParameters, StandardState, StandardModel


class RTBMParameters(StandardParameters):
    """ Parameters of the RTBM. """
    par_wm: float = 4  # Asymmetry factor for negative utilities
    gamma: float = 0.3  # Sensitivity exponent of the generalized utility
    tau: float = 5  # Maximum anticipation time horizon
    alpha: float = 0.08  # Speed uncertainty variation coefficient
    par_wc: float = 100000  # Weighing factor for accidents


class RTBMState(StandardState):
    """ State of the RTBM. """


class RTBM(StandardModel):
    """ Class for simulation of the Risk-Taking Behavior Model (RTBM). """
    def __init__(self):
        StandardModel.__init__(self)
        self.parms = RTBMParameters()
        self.state = RTBMState()

    def init_simulation(self, parms: RTBMParameters) -> None:
        """ Initialize the simulation.

        See the StandardModel for the default parameters.
        The following additional parameters can also be set:
        - par_wm: float = 4  # Asymmetry factor for negative utilities
        - gamma: float = 0.3  # Sensitivity exponent of the generalized utility
        - tau: float = 5  # Maximum anticipation time horizon
        - alpha: float = 0.08  # Speed uncertainty variation coefficient
        - par_wc: float = 100000  # Weighing factor for accidents

        :param parms: The parameters listed above.
        """
        # Set RTBM parameters.
        self.parms.par_wm, self.parms.par_wc = parms.par_wm, parms.par_wc
        self.parms.gamma, self.parms.tau, self.parms.alpha = parms.gamma, parms.tau, parms.alpha

        StandardModel.init_simulation(self, parms)

    def acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        accelerations = np.concatenate((np.linspace(self.parms.amin, 0, 100, endpoint=False),
                                        np.linspace(self.parms.amin, 2, 21)))
        return accelerations[np.argmax(self._utility(accelerations, gap, vhost, vdiff))]

    def _upt(self, acceleration):
        # prospect-theoretic acceleration utility
        return (acceleration *
                (self.parms.par_wm + .5*(1-self.parms.par_wm)*(np.tanh(acceleration)+1)) *
                (1 + acceleration**2)**(.5*(self.parms.gamma-1)))

    def _pcollision(self, acceleration, distance, vhost, vdiff):
        vlead = vhost - vdiff
        return norm.cdf((vdiff - distance/self.parms.tau + .5*acceleration*self.parms.tau) /
                        (self.parms.alpha * vlead))

    def _utility(self, acceleration, distance, vhost, vdiff):
        pcol = self._pcollision(acceleration, distance, vhost, vdiff)
        return self._upt(acceleration) - pcol*self.parms.par_wc
