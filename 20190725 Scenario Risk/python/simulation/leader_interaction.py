""" Modeling of a leader vehicle that approaches a constant speed after a while

Creation date: 2020 10 08
Author(s): Erwin de Gelder

Modifications:
"""

import numpy as np
from .options import Options


class LeaderInteractionParameters(Options):
    """ Parameters for the lead vehicle. """
    init_position: float = 0
    init_speed: float = 1
    speed_difference: float = 0
    init_acceleration: float = 0
    duration: float = 10


class LeaderInteractionState(Options):
    """ State of the lead vehicle. """
    position: float = 0
    speed: float = 0
    acceleration: float = 0


class LeaderInteraction:
    """ Vehicle that approaches a constant speed after a while.

    The speed profile follows a polynomial.
    """
    def __init__(self):
        self.state = LeaderInteractionState()
        self.parms = LeaderInteractionParameters()
        self.polycoefficients = (0, 0, 0, 1)
        self.polycoefficients_der = (0, 0, 1)
        self.polycoefficients_int = (0, 0, 0, 0, 1)

    def init_simulation(self, parms: LeaderInteractionParameters) -> None:
        """ Initialize the simulation.

        The following parameters can be set:
        init_position: float = 0
        init_speed: float = 1
        speed_difference: float = 0
        init_acceleration: float = 0
        duration: float = 10

        :param parms: The parameters listed above.
        """
        self.parms.init_position = parms.init_position
        self.parms.init_speed = parms.init_speed
        self.parms.speed_difference = parms.speed_difference
        self.parms.init_acceleration = parms.init_acceleration
        self.parms.duration = parms.duration
        self.polycoefficients = (((parms.init_acceleration*parms.duration -
                                   2*parms.speed_difference) /
                                  parms.duration**3),
                                 ((3*parms.speed_difference -
                                   2*parms.init_acceleration*parms.duration) /
                                  parms.duration**2),
                                 parms.init_acceleration,
                                 parms.init_speed)
        self.polycoefficients_der = (3*self.polycoefficients[0],
                                     2*self.polycoefficients[1],
                                     self.polycoefficients[2])
        self.polycoefficients_int = (self.polycoefficients[0] / 4,
                                     self.polycoefficients[1] / 3,
                                     self.polycoefficients[2] / 2,
                                     self.polycoefficients[3],
                                     parms.init_position)

    def step_simulation(self, time: float) -> None:
        """ Compute the state (position, speed) at time t.

        :param time: The time of the simulation.
        """
        if time < self.parms.duration:
            self.state.acceleration = np.polyval(self.polycoefficients_der, time)
            self.state.speed = np.polyval(self.polycoefficients, time)
            self.state.position = np.polyval(self.polycoefficients_int, time)
        else:
            self.state.acceleration = 0
            self.state.speed = self.parms.init_speed + self.parms.speed_difference
            self.state.position = (np.polyval(self.polycoefficients_int, self.parms.duration) +
                                   (self.parms.init_speed + self.parms.speed_difference) *
                                   (time - self.parms.duration))
