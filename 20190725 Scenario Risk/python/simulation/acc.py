""" Model of Adaptive Cruise Control (ACC) from Xiao et al. (2017).

Creation date: 2020 08 05
Author(s): Erwin de Gelder

Modifications:
2018 08 13 To compute the acceleration, take the minimum of the CC and the ACC.
"""

from .standard_model import StandardParameters, StandardState, StandardModel


class ACCParameters(StandardParameters):
    """ Parameters for the HDM. """
    k_cruise: float = 0.4  # [s^-1] Cruising gain
    k1_acc: float = 0.23  # [s^-2] Spacing error gain
    k2_acc: float = 0.07  # [s^-1] Speed difference gain
    thw: float = 1.1  # [s] Desired time headway (in StandardParameters, thw=1.0)
    sensor_range: float = 150  # [m] Maximum range to see a target vehicle


class ACCState(StandardState):
    """ State of the ACC. """


class ACC(StandardModel):
    """ Class for simulation of an ACC. """
    def __init__(self):
        StandardModel.__init__(self)
        self.parms = ACCParameters()
        self.state = ACCState()

    def init_simulation(self, parms: ACCParameters) -> None:
        """ Initialize the simulation.

        See the StandardModel for the default parameters.
        The following additional parameters can also be set:
        k_cruise: float = 0.4  # [s^-1] Cruising gain
        k1_acc: float = 0.23  # [s^-2] Spacing error gain
        k2_acc: float = 0.07  # [s^-1] Speed difference gain
        sensor_range: float = 150  # [m] Maximum range to see a target vehicle

        Note that the default value for thw is 1.1 s (as opposed to the 1.0 in
        StandardParameters.

        :param parms: The parameters listed above.
        """
        # Set parameters.
        self.parms.k_cruise, self.parms.sensor_range = parms.k_cruise, parms.sensor_range
        self.parms.k1_acc, parms.k2_acc = parms.k1_acc, parms.k2_acc

        StandardModel.init_simulation(self, parms)

    def acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        """ Compute the acceleration based on the gap, vhost, vdiff.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        :return: The acceleration.
        """
        # If target is out of range, use the cruise control.
        if gap > self.parms.sensor_range or (gap < 0 and self.parms.cruise_after_collision):
            return self._acceleration_cc(vhost)
        return min(self._acceleration_cc(vhost),
                   self._acceleration_acc(gap, vhost, vdiff))

    def _acceleration_cc(self, vhost: float) -> float:
        return 0.0  # self.parms.k_cruise * (self.parms.speed - vhost)

    def _acceleration_acc(self, gap: float, vhost: float, vdiff: float) -> float:
        safety_distance = self.safety_distance(vhost)
        error_distance = gap - safety_distance - self.parms.thw*vhost
        return self.parms.k1_acc*error_distance - self.parms.k2_acc*vdiff

    @staticmethod
    def safety_distance(vhost: float) -> float:
        if vhost >= 15:
            return 5
        if vhost >= 10.8:
            return 75 / vhost
        return 7
