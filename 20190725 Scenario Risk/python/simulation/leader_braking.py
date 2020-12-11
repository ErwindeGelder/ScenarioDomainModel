""" Modeling of a leader vehicle that starts braking after a fixed time.

Creation date: 2020 05 27
Author(s): Erwin de Gelder

Modifications:
2020 06 24 Do not return speed and position with an update step.
"""

import numpy as np
from .options import Options


class LeaderBrakingParameters(Options):
    """ Parameters for the braking lead vehicle. """
    init_position: float = 0
    init_speed: float = 1
    speed_difference: float = 1
    average_deceleration: float = 1
    duration: float = None
    tconst: float = 5


class LeaderBrakingState(Options):
    """ State of the braking lead vehicle. """
    position: float = 0
    speed: float = 0
    acceleration: float = 0


class LeaderBraking:
    """ Vehicle that starts braking after a fixed time.

    The braking activity follows a sinusoidal function. Before and after the
    braking, the speed is constant.
    """
    def __init__(self):
        self.state = LeaderBrakingState()
        self.parms = LeaderBrakingParameters()

    def init_simulation(self, parms: LeaderBrakingParameters) -> None:
        """ Initialize the simulation.

        The following parameters can be set:
        - speed_difference
        - average_deceleration
        - duration (if not given, calculated from the two previous parameters)
        - init_position
        - init_speed

        :param parms: The parameters listed above.
        """
        self.parms.speed_difference = parms.speed_difference
        if parms.duration is None:
            self.parms.duration = parms.speed_difference / parms.average_deceleration
        else:
            self.parms.duration = parms.duration
        self.parms.init_position = parms.init_position
        self.parms.init_speed = parms.init_speed
        self.parms.tconst = parms.tconst

        self.state.speed = self.parms.init_speed
        self.state.position = self.parms.init_position

    def step_simulation(self, time: float) -> None:
        """ Compute the state (position, speed) at time t.

        :param time: The time of the simulation.
        """
        if time <= self.parms.tconst:
            acceleration = 0
            speed = self.parms.init_speed
            distance = self.parms.init_position + self.parms.init_speed * time
        elif time < self.parms.tconst + self.parms.duration:
            acceleration = -np.pi*self.parms.speed_difference / self.parms.duration * \
                np.sin(np.pi*(time-self.parms.tconst)/self.parms.duration) / 2
            speed = self.parms.init_speed - self.parms.speed_difference/2 * \
                (1-np.cos(np.pi*(time-self.parms.tconst)/self.parms.duration))
            distance = self.parms.init_position + self.parms.init_speed * time - \
                self.parms.speed_difference/2*(time-self.parms.tconst-self.parms.duration/np.pi *
                                               np.sin(np.pi*(time-self.parms.tconst) /
                                                      self.parms.duration))
        else:
            acceleration = 0
            speed = self.parms.init_speed - self.parms.speed_difference
            distance = self.parms.init_position + \
                self.parms.speed_difference * (self.parms.duration / 2 + self.parms.tconst) + \
                (self.parms.init_speed - self.parms.speed_difference) * time

        self.state.position = distance
        self.state.speed = speed
        self.state.acceleration = acceleration
