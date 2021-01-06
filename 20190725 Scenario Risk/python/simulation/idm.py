""" Intelligent Driver Model (IDM) from Treiber et al. (2000).

Creation date: 2020 05 27
Author(s): Erwin de Gelder

Modifications:
2020 06 12 Avoid division by zero when calculating the non-free-flow part.
2020 06 22 A seperate function for the integration of the acceleration.
2020 06 24 Do not return speed and position with an update step.
2020 08 05 Make use of the StandardModel.
2020 08 12 Add options of having a maximum view. Targets further away are not considered.
"""

import numpy as np
from .standard_model import StandardParameters, StandardState, StandardModel


class IDMParameters(StandardParameters):
    """ Parameters for the IDM. """
    # pylint: disable=too-many-instance-attributes
    a_acc: float = 0.73  # IDM parameter (preferred maximum acceleration)
    b_acc: float = 1.67  # IDM parameter (preferred minimum acceleration)
    delta: float = 4  # IDM parameter
    safety_distance: float = 2  # IDM parameter (safety distance)
    max_view: float = 150  # [m] Maximum distance to see a target


class IDMState(StandardState):
    """ State of the IDM. """


class IDM(StandardModel):
    """ Class for simulation of the Intelligent Driver Model (IDM). """
    def __init__(self):
        StandardModel.__init__(self)
        self.parms = IDMParameters()
        self.state = IDMState()

    def init_simulation(self, parms: IDMParameters) -> None:
        """ Initialize the simulation.

        See the StandardModel for the default parameters.
        The following additional parameters can also be set:
        - a_acc: float = 0.73  # IDM parameter (preferred maximum acceleration)
        - b_acc: float = 1.67  # IDM parameter (preferred minimum acceleration)
        - delta: float = 4  # IDM parameter
        - safety_distance: float = 2  # IDM parameter (safety distance)
        - max_view: float = 150  # [m] Maximum distance to see a target

        :param parms: The parameters listed above.
        """
        # Set IDM parameters.
        self.parms.a_acc, self.parms.b_acc = parms.a_acc, parms.b_acc
        self.parms.delta = parms.delta
        self.parms.safety_distance = parms.safety_distance
        self.parms.max_view = parms.max_view

        StandardModel.init_simulation(self, parms)

    def acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        """ Compute the acceleration based on the gap, vhost, vdiff.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        :return: The acceleration.
        """
        if gap > self.parms.max_view:
            return self.parms.a_acc * (1 - self._freeflowpart(vhost))
        return self.parms.a_acc * (1 - self._freeflowpart(vhost) -
                                   self._nonfreeflowpart(gap, vhost, vdiff))

    def _freeflowpart(self, vhost: float) -> float:
        """ Compute the part of the IDM that does not consider the lead vehicle.

        :param vhost: Speed of host vehicle.
        :return: The contribution of the free flow part.
        """
        return (vhost/self.parms.speed)**self.parms.delta

    def _nonfreeflowpart(self, gap: float, vhost: float, vdiff: float) -> float:
        """ Compute the quantity of the IDM that considers the lead vehicle.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        :return: The contribution of the non-free flow part.
        """
        if gap == 0:
            return self.parms.amin
        sstar = (self.parms.safety_distance +
                 self.parms.thw * vhost +
                 vhost * vdiff / (2 * np.sqrt(self.parms.a_acc * self.parms.b_acc)))
        return (sstar / gap) ** 2
