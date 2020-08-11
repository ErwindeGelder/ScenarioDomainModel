""" Model of Cooperative Adaptive Cruise Control (CACC) from Xiao et al. (2017).

Creation date: 2020 08 10
Author(s): Erwin de Gelder

CACC is not yet included, because it behaves not correct. The main problem is
that the original model directly updates the speed, such that the acceleration
is undefined.

Modifications:
"""

from .acc import ACC
from .standard_model import StandardParameters, StandardState, StandardModel


class CACCParameters(StandardParameters):
    """ Parameters for the HDM. """
    k_cruise: float = 0.4  # [s^-1] Cruising gain
    kp_cacc: float = 0.45  # [s^-1] Proportional gain
    kd_cacc: float = 0.25  # [-] Derivetive gain
    thw: float = 0.6  # [s] Desired time headway (in StandardParameters, thw=1.0)
    sensor_range: float = 300  # [m] Maximum range to see a target vehicle


class CACCState(StandardState):
    """ State of the CACC. """


class CACC(ACC):
    """ Class for simulation of a CACC. """
    def __init__(self):
        ACC.__init__(self)
        self.parms = CACCParameters()
        self.state = CACCState()

    def init_simulation(self, parms: CACCParameters) -> None:
        """ Initialize the simulation.

        See the StandardModel for the default parameters.
        The following additional parameters can also be set:
        k_cruise: float = 0.4  # [s^-1] Cruising gain
        k1_acc: float = 0.23  # [s^-2] Spacing error gain
        k2_acc: float = 0.07  # [s^-1] Speed difference gain
        sensor_range: float = 300  # [m] Maximum range to see a target vehicle

        Note that the default value for thw is 1.1 s (as opposed to the 1.0 in
        StandardParameters.

        :param parms: The parameters listed above.
        """
        # Set parameters.
        self.parms.k_cruise, self.parms.sensor_range = parms.k_cruise, parms.sensor_range
        self.parms.kp_cacc, parms.kd_cacc = parms.kp_cacc, parms.kd_cacc

        StandardModel.init_simulation(self, parms)

    def acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        """ Compute the acceleration based on the gap, vhost, vdiff.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        :return: The acceleration.
        """
        # If target is out of range, use the cruise control.
        if gap > self.parms.sensor_range:
            return self._acceleration_cc(vhost)
        return self._acceleration_cacc(gap, vhost, vdiff)

    def _acceleration_cacc(self, gap: float, vhost: float, vdiff: float) -> float:
        safety_distance = self.safety_distance(vhost)
        error_distance = gap - safety_distance - self.parms.thw*vhost
        safety_distance_dot = self.safety_distance_dot(vhost)
        error_distance_dot = -vdiff - safety_distance_dot - self.parms.thw*self.state.acceleration
        vdiff = self.parms.kp_cacc*error_distance + self.parms.kd_cacc*error_distance_dot
        return vdiff / self.parms.timestep

    @staticmethod
    def safety_distance(vhost: float) -> float:
        if vhost >= 10:
            return 5
        return -0.125*vhost + 6.25

    def safety_distance_dot(self, vhost: float) -> float:
        if vhost >= 10:
            return 0
        return -0.125*self.state.acceleration
