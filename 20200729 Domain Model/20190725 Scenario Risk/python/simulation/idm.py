""" Intelligent Driver Model (IDM) from Treiber et al. (2000).

Creation date: 2020 05 27
Author(s): Erwin de Gelder

Modifications:
2020 06 12 Avoid division by zero when calculating the non-free-flow part.
2020 06 22 A seperate function for the integration of the acceleration.
2020 06 24 Do not return speed and position with an update step.
"""

import collections
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
    acceleration: float = 0


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
        - thw: float = 1  # IDM parameter (time headway)
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
        self.state.acceleration = 0

    def step_simulation(self, leader) -> None:
        """ Compute the state (position, speed).

        :param leader: The leading vehicle that contains position and speed.
        """
        self.update(leader.state.position - self.state.position,
                    self.state.speed,
                    self.state.speed - leader.state.speed)

    def update(self, gap: float, vhost: float, vdiff: float) -> None:
        """ Compute a step using the inputs as stated in Treiber et al. (2006).

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        """
        self.integration_step()

        # Calculate acceleration based on IDM
        self.accelerations.append(self._acceleration(gap, vhost, vdiff))

    def integration_step(self) -> None:
        """ Integrate the acceleration to obtain speed and position.

        Because the state will be updated, there is nothing to return.
        """
        # Update speed
        self.state.acceleration = np.max((self.parms.amin, self.accelerations[0]))
        self.state.speed += self.state.acceleration * self.parms.dt

        # Update position
        self.state.position += self.state.speed * self.parms.dt

    def _acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        """ Compute the acceleration.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        :return: The acceleration.
        """
        return self.parms.a_acc * (1 - self._freeflowpart(vhost) -
                                   self._nonfreeflowpart(gap, vhost, vdiff))

    def _freeflowpart(self, vhost: float) -> float:
        """ Compute the part of the IDM that does not consider the lead vehicle.

        :param vhost: Speed of host vehicle.
        :return: The contribution of the free flow part.
        """
        return (vhost / self.parms.free_speed) ** self.parms.delta

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
