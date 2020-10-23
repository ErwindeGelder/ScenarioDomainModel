""" Class that does not allow for creating new attributes.

Creation date: 2020 10 22
Author(s): Erwin de Gelder

Modifications:
"""

from typing import Union
import numpy as np
from scipy.special import erf  # pylint: disable=no-name-in-module
from .options import Options


class WangStamatiadisParameters(Options):
    """ Parameters for the risk metric of Wang & Stamatiadis. """
    lower_limit_braking_rate: float = 4.23  # [m/s2]
    upper_limit_braking_rate: float = 12.68  # [m/s2]
    mean_braking_rate: float = 8.45  # [m/s2]
    std_braking_rate: float = 1.40  # [m/s2]
    n_braking_rate: int = 100  # Number of samples to approximate integral

    normal_mu_reaction_time: float = 0.92  # [s] mean of the samples from the log-normal dist.
    normal_sigma_reaction_time: float = 0.28  # [s] standard deviation of samples of log-normal
    mu_reaction_time: float = None  # It will be calculated if not given.
    sigma_reaction_time: float = None  # It will be calculated if not given.


class WangStamatiadis:
    """ Calculation of the CPM as proposed by Wang & Stamatiadis

    CPM stands for Crash Propensity Metric.

    Attributes:
    parms(WangStamatiadisParameters): parameters that need to be set during
        initialization.
    vec_braking_rate(np.ndarray): Array with braking rate values used for the
        integral.
    prob_density_braking_rate(np.ndarray): Corresponding probability densities
        of the braking rate.
    d_braking_rate(float): Width of one data point - used for integration.
    """

    def __init__(self, parameters: WangStamatiadisParameters = None):
        self.parms = WangStamatiadisParameters() if parameters is None else parameters

        # Compute the probability density for the braking rate.
        self.vec_braking_rate = np.linspace(self.parms.lower_limit_braking_rate,
                                            self.parms.upper_limit_braking_rate,
                                            self.parms.n_braking_rate, endpoint=False)
        self.d_braking_rate = self.vec_braking_rate[1]-self.vec_braking_rate[0]
        self.vec_braking_rate += self.d_braking_rate/2
        self.prob_density_braking_rate = \
            (np.exp(-(self.vec_braking_rate-self.parms.mean_braking_rate)**2/
                    (2*self.parms.std_braking_rate**2))/
             (np.sqrt(2*np.pi)*self.parms.std_braking_rate))
        self.prob_density_braking_rate /= np.sum(self.prob_density_braking_rate)*self.d_braking_rate

        # Calculate mu for the log-normal distribution.
        if self.parms.mu_reaction_time is None:
            self.parms.mu_reaction_time = np.log(self.parms.normal_mu_reaction_time**2 /
                                                 np.sqrt(self.parms.normal_mu_reaction_time**2 +
                                                         self.parms.normal_sigma_reaction_time**2))
        if self.parms.sigma_reaction_time is None:
            self.parms.sigma_reaction_time = np.sqrt(np.log(
                1 + self.parms.normal_sigma_reaction_time**2/self.parms.normal_mu_reaction_time**2))

    def cdf_reaction_time(self, reaction_time: Union[float, np.ndarray]) -> float:
        """ Calculate the cumulative distribution function of the reaction time.

        In other words, we calculate the probability that the actual reaction
        time of a driver is less that the provided time.

        :param reaction_time: Value for which the CDF is to be evaluated.
        :return: The CDF of the reaction time.
        """
        return (.5+.5*erf((np.log(reaction_time)-self.parms.mu_reaction_time)/
                          (np.sqrt(2)*self.parms.sigma_reaction_time)))

    @staticmethod
    def required_reaction_time(ttc: float, speed_diff: float,
                               braking_rate: Union[float, np.ndarray]) -> float:
        """ Calculate the maximum reaction time to avoid a collision.

        The time is based on the assumption that the lead vehicle drives as a
        constant speed and the ego vehicle decelerates after the reaction time
        with a deceleration equal to `braking_rate`.

        :param ttc: The initial time-to-collision.
        :param speed_diff: The initial speed difference.
        :param braking_rate: The deceleration the ego vehicle applies.
        :return: The maximum reaction to to avoid a collision.
        """
        return ttc-speed_diff/(2*braking_rate)

    def prob_no_collision(self, ttc: float, speed_diff: float) -> float:
        """ Calculate the probability of avoiding a collision.

        :param ttc: The current time-to-collision.
        :param speed_diff: The current speed difference.
        :return: Probability of avoiding a collision.
        """
        # If the speed difference is negative, the probability of no collsion is 1.
        if speed_diff <= 0.0:
            return 1.0

        # If the TTC is negative while the speed difference is positive, we have a collision!
        if ttc <= 0.0:
            return 0.0

        # Ensure that the lower of the integral is not smaller than speed_diff/2ttc.
        # This would result in a logarithmic of a negative value.
        if speed_diff/(2*ttc) <= self.vec_braking_rate[0]:
            i_min = 0
        else:
            # If speed_diff/2ttc is larger than the upper limit, simply return 0, as
            # there is no way to avoid the collision.
            lower_limit = speed_diff/(2*ttc)
            if lower_limit >= self.vec_braking_rate[-1]:
                return 0.0
            i_min = next((i for i, braking_rate in enumerate(self.vec_braking_rate)
                          if braking_rate > lower_limit), )

        # Take the integral.
        required_reaction_time = self.required_reaction_time(ttc, speed_diff,
                                                             self.vec_braking_rate[i_min:])
        cdf_reaction_time = self.cdf_reaction_time(required_reaction_time)
        integral = (np.sum(self.prob_density_braking_rate[i_min:] * cdf_reaction_time) *
                    self.d_braking_rate)
        # Ensure that rounding errors do not lead to a probability larger than 1.0
        return min(integral, 1.0)

    def prob_collision(self, ttc: float, speed_diff: float) -> float:
        """ Calculate the probability of a collision.

        :param ttc: The current time-to-collision.
        :param speed_diff: The current speed difference.
        :return: Probability of a collision.
        """
        return 1 - self.prob_no_collision(ttc, speed_diff)


if __name__ == "__main__":
    WS = WangStamatiadis()
    for TTC, SPEEDDIFF in zip([1, 1, 1, 2, 2, 2, 3, 3, 3],
                              [5, 10, 15, 5, 10, 15, 10, 15, 20]):
        print("TTC: {:.1f}, Speed difference: {:2.0f}, Collision probability: {:.3f}"
              .format(TTC, SPEEDDIFF, WS.prob_collision(TTC, SPEEDDIFF)))
