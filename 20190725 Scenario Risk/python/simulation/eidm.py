""" Enhanced Intelligent Driver Model (IDM) (aka ideal ACC) from Kesting et al. (2010).

Creation date: 2020 06 22
Author(s): Erwin de Gelder

Modifications:
2020 06 24 Do not return speed and position with an update step.
"""

import numpy as np
from .idm import IDM, IDMParameters, IDMState


class EIDMParameters(IDMParameters):
    """ Parameters for the IDM. """
    # pylint: disable=too-many-instance-attributes
    coolness = 0.99

    def __init__(self, **kwargs):
        IDMParameters.__init__(self, **kwargs)


class EIDMState(IDMState):
    """ State of the EIDM. """
    speed_leader: float = 0

    def __init__(self):
        IDMState.__init__(self)


class EIDM(IDM):
    """ Class for simulation of the Intelligent Driver Model (IDM). """
    def __init__(self):
        IDM.__init__(self)
        self.state = EIDMState()
        self.parms = EIDMParameters()

    def init_simulation(self, parms: EIDMParameters) -> None:
        """ Initialize the simulation.

        :param parms: The parameters listed above.
        """
        IDM.init_simulation(self, parms)

        # Set additional parameter.
        self.parms.coolness = parms.coolness

        # Set speed of leading vehicle at None, because it is unknown.
        self.state.speed_leader = 0

    def step_simulation(self, leader) -> None:
        """ Compute the state (position, speed).

        :param leader: The leading vehicle that contains position, speed, and
                       acceleration.
        """
        self.update_eidm(leader.state.position - self.state.position,
                         self.state.speed,
                         self.state.speed - leader.state.speed,
                         self.state.acceleration)

    def update_eidm(self, gap: float, vhost: float, vdiff: float, alead: float) -> None:
        """ Compute a step using the inputs as stated in Kesting et al. (2010).

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        :param alead: The acceleration of the leading vehicle.
        """
        self.integration_step()

        # In case of a negative gap, brake as much as possible.
        if gap <= 0:
            self.accelerations.append(min(-10., self.parms.amin))
            return

        # Calculate acceleration based on IDM.
        a_idm = self.acceleration(gap, vhost, vdiff)

        # Calculate acceleration according to the Constant-Acceleration Heuristic (CAH).
        alead = min(alead, self.state.acceleration)
        if self.state.speed_leader*vdiff < -2*gap*alead:
            a_cah = vhost**2 * alead / (vhost**2 - 2*gap*alead)
        else:
            a_cah = alead
            if vdiff > 0:
                a_cah -= vdiff*2 / (2*gap)

        # Calculate the acceleration of the EIDM.
        if a_idm >= a_cah:
            a_acc = a_idm
        else:
            a_acc = ((1-self.parms.coolness)*a_idm + self.parms.coolness *
                     (a_cah+self.parms.b_acc*np.tanh((a_idm - a_cah) / self.parms.b_acc)))
        self.accelerations.append(a_acc)
