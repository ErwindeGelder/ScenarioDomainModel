""" Intelligent Driver Model (IDM) from Treiber et al. (2000).

Creation date: 2020 05 27
Author(s): Erwin de Gelder

Modifications:
"""

import collections
from typing import Tuple
import numpy as np
from .options import Options


class IDMParameters(Options):
    """ Parameters for the IDM. """
    # pylint: disable=too-many-instance-attributes
    a_acc: float = 0.73  # IDM parameter (preferred maximum acceleration)
    b_acc: float = 1.67  # IDM parameter (preferred minimum acceleration)
    delta: float = 4  # IDM parameter
    free_speed: float = 30  # IDM parameter (free flow speed)
    safety_distance: float = 2  # IDM parameter (safety distance)
    thw: float = 1  # IDM parameter (time headway)
    amin: float = -np.inf  # Custom parameter (minimum acceleration)
    n_reaction: int = 0  # Custom parameter (samples of delay)
    dt: float = 0.01  # Sample time (needed for delay)
    init_position: float = 0
    init_speed: float = 1


class IDMState(Options):
    """ State of the IDM. """
    position: float = 0
    speed: float = 0


class IDM:
    """ Class for simulation of the Intelligent Driver Model (IDM). """
    def __init__(self):
        self.parms = IDMParameters()
        self.state = IDMState()
        self.accelerations = collections.deque(maxlen=1)

    def init_simulation(self, parms: IDMParameters) -> None:
        """ Initialize the simulation.

        The following parameters can be set:
        - a: float = 0.73  # IDM parameter (preferred maximum acceleration)
        - b: float = 1.67  # IDM parameter (preferred minimum acceleration)
        - delta: float = 4  # IDM parameter
        - v0: float = 30  # IDM parameter (free flow speed)
        - s0: float = 2  # IDM parameter (safety distance)
        - T: float = 1  # IDM parameter (time headway)
        - amin: float = -np.inf  # Custom parameter (minimum acceleration)
        - tr: int = 0  # Custom parameter (samples of delay)
        - dt: float = 0.01  # Sample time (needed for delay)
        - init_position
        - init_speed

        :param parms: The parameters listed above.
        """
        # Set parameters.
        self.parms.a_acc, self.parms.b_acc = parms.a_acc, parms.b_acc
        self.parms.free_speed, self.parms.delta = parms.free_speed, parms.delta
        self.parms.safety_distance, self.parms.thw = parms.safety_distance, parms.thw
        self.parms.amin, self.parms.n_reaction = parms.amin, parms.n_reaction

        # Create the list with accelerations to account for the delay.
        self.accelerations = collections.deque(maxlen=self.parms.n_reaction+1)
        self.accelerations.append(0)

        # Set state.
        self.state.position = parms.init_position
        self.state.speed = parms.init_speed

    def step_simulation(self, xlead: float, vlead: float) -> Tuple[float, float]:
        """ Compute the state (position, speed).

        :param xlead: Position of leading vehicle.
        :param vlead: Speed of leading vehicle.
        :return: Position and speed.
        """
        # Update speed
        acceleration = np.max((self.parms.amin, self.accelerations[0]))
        self.state.speed += acceleration * self.parms.dt

        # Update position
        self.state.position += self.state.speed * self.parms.dt

        # Calculate acceleration based on IDM
        self.accelerations.append(self._acceleration(xlead, vlead))

        return self.state.position, self.state.speed

    def _acceleration(self, xlead: float, vlead: float) -> float:
        """ Compute the acceleration.

        :param xlead: Position of leading vehicle.
        :param vlead: Speed of leading vehicle.
        :return: The acceleration.
        """
        return self.parms.a_acc * (1 - self._freeflowpart() - self._nonfreeflowpart(xlead, vlead))

    def _freeflowpart(self) -> float:
        """ Compute the part of the IDM that does not consider the lead vehicle.

        :return: The contribution of the free flow part.
        """
        return (self.state.speed / self.parms.free_speed) ** self.parms.delta

    def _nonfreeflowpart(self, xlead: float, vlead: float) -> float:
        """ Compute the quantity of the IDM that considers the lead vehicle.

        :param vlead: Speed of leading vehicle.
        :return: s*.
        """
        sstar = (self.parms.safety_distance +
                 self.parms.thw * self.state.speed +
                 self.state.speed * (self.state.speed-vlead) /
                 (2 * np.sqrt(self.parms.a_acc * self.parms.b_acc)))
        return (sstar / (xlead - self.state.position)) ** 2
