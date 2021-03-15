""" Modeling of a leader vehicle that approaches a constant speed after a while

Creation date: 2020 10 08
Author(s): Erwin de Gelder

Modifications:
2020 10 28: Make sure that final speed is not negative.
"""

import numpy as np
from .options import Options


class LeaderInteractionParameters(Options):
    """ Parameters for the lead vehicle. """
    init_position: float = 0
    init_speed: float = 1
    velocities: np.ndarray = np.array([])
    times: np.ndarray = np.array([])
    time_to_smooth_acceleration: float = 2


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
        self.velocity_profile = np.array([])
        self.lasttime = 0
        self.smooth_end_acceleration = 2  # [seconds] used to smooth the acceleration

    def init_simulation(self, parms: LeaderInteractionParameters) -> None:
        """ Initialize the simulation.

        The following parameters can be set:
        init_position: float = 0
        init_speed: float = 1
        times: np.ndarray
        velocities: np.ndarray

        :param parms: The parameters listed above.
        """
        self.parms.init_position = parms.init_position
        self.parms.init_speed = parms.init_speed
        self.lasttime = 0
        final_acceleration = ((parms.velocities[-1] - parms.velocities[-2]) /
                              (parms.times[-1] - parms.times[-2]))
        smooth_times = np.linspace(0, parms.time_to_smooth_acceleration, 50)
        smooth_velocities = (-final_acceleration/4 * smooth_times**2 +
                             final_acceleration * smooth_times +
                             parms.velocities[-1])
        self.parms.times = np.concatenate((parms.times, smooth_times[1:]+parms.times[-1], [1e9]))
        self.parms.velocities = np.concatenate((parms.velocities, smooth_velocities[1:],
                                                [smooth_velocities[-1]]))
        self.parms.velocities = self.parms.velocities.clip(min=0)

        self.state.position = self.parms.init_position
        self.state.speed = self.parms.init_speed

    def step_simulation(self, time: float) -> None:
        """ Compute the state (position, speed) at time t.

        :param time: The time of the simulation.
        """
        new_speed = np.interp(time, self.parms.times, self.parms.velocities)
        self.state.acceleration = (new_speed - self.state.speed) / (time - self.lasttime)
        self.state.speed = new_speed
        self.state.position += self.state.speed * (time - self.lasttime)
        self.lasttime = time
