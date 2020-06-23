""" Human Driver (Meta-)Model (HDM) from Treiber et al. (2006).

Creation date: 2020 05 27
Author(s): Erwin de Gelder

Modifications:
2020 06 23 Provide the parameters of the underlying model as part of the HDM parameters. Add
           position, speed, and acceleration to own state.
"""

from typing import Tuple, Union
import numpy as np
from .options import Options
from .idm import IDM, IDMParameters, IDMState
from .idmplus import IDMPlus


class HDMParameters(Options):
    """ Parameters for the HDM. """
    model: Union[IDM, IDMPlus] = None
    parms_model: IDMParameters = None
    speed_std: float = 0.05
    tau: float = 20
    rttc: float = 0.01  # Estimation error of reciprocal ttc
    dt: float = 0.01  # Sample time (needed for Wiener process)
    decay: float = None  # Calculated based on dt and tau
    contribution_noise: float = None  # Calculated based on dt and tau
    delay: float = None  # Calculated based on dt and n_reaction (from model parameters)


class HDMState(IDMState):
    """ State of the HDM. """
    w_gap: float = 0
    w_speed: float = 0

    def __init__(self, **kwargs):
        IDMState.__init__(self, **kwargs)


class HDM:
    """ Class for simulation of the Intelligent Driver Model (IDM). """
    def __init__(self):
        self.parms = HDMParameters()
        self.state = HDMState()

    def init_simulation(self, parms: HDMParameters) -> None:
        """ Initialize the simulation.

        The following parameters can be set:
        - model
        - speed_std
        - tau
        - rttc

        :param parms: The parameters listed above.
        """
        # Set parameters.
        self.parms.model = parms.model
        self.parms.speed_std = parms.speed_std
        self.parms.tau = parms.tau
        self.parms.rttc = parms.rttc
        self.parms.contribution_noise = np.sqrt(2*self.parms.dt / self.parms.tau)
        self.parms.decay = np.exp(-self.parms.dt / self.parms.tau)
        self.parms.model.init_simulation(parms.parms_model)
        self.parms.delay = self.parms.dt * self.parms.model.parms.n_reaction
        self.state.w_speed = 0
        self.state.w_gap = 0
        self.state.speed = self.parms.model.state.speed
        self.state.position = self.parms.model.state.position
        self.state.acceleration = self.parms.model.accelerations

    def step_simulation(self, xlead: float, vlead: float) -> Tuple[float, float]:
        """ Compute the state (position, speed).

        This will call the update function of the model, using the inputs
        calculated by self._temporal_anticipation.

        :param xlead: Position of leading vehicle.
        :param vlead: Speed of leading vehicle.
        :return: Position and speed.
        """
        gap_est, vlead = self._estimate_gap_speed(xlead, vlead)
        gap, vhost, vdiff = self._temporal_anticipation(gap_est, vlead)
        position, speed = self.parms.model.update(gap, vhost, vdiff)
        self.state.position = self.parms.model.state.position
        self.state.speed = self.parms.model.state.speed
        self.state.acceleration = self.parms.model.state.acceleration
        return position, speed

    def _update_wiener(self) -> None:
        """ Do an update of the Wiener process. """
        self.state.w_speed = (self.parms.decay*self.state.w_speed +
                              self.parms.contribution_noise*np.random.randn())
        self.state.w_gap = (self.parms.decay*self.state.w_gap +
                            self.parms.contribution_noise*np.random.randn())

    def _estimate_gap_speed(self, xlead: float, vlead: float) -> Tuple[float, float]:
        """ Estimate the real gap and speed of lead vehicle.

        :param xlead: Position of leading vehicle.
        :param vlead: Speed of leading vehicle.
        :return: Estimated gap and speed
        """
        self._update_wiener()
        gap = xlead - self.parms.model.state.position
        gap_est = gap*np.exp(self.parms.speed_std * self.state.w_gap)
        vlead_est = vlead + gap*self.parms.rttc*self.state.w_speed
        return gap_est, vlead_est

    def _temporal_anticipation(self, gap_est: float, vlead_est: float) -> \
            Tuple[float, float, float]:
        """ Correct the 'measurements' based on the anticipated delay.

        :param gap_est: Estimated gap.
        :param vlead_est: Estimated speed of lead vehicle.
        :return: Anticipated gap, speed host vehicle, and speed difference with lead.
        """
        gap_corr = gap_est - self.parms.delay*(vlead_est - self.parms.model.state.speed)
        speed_corr = (self.parms.model.state.speed +
                      self.parms.delay*self.parms.model.state.acceleration)
        speed_diff_corr = self.parms.model.state.speed - vlead_est
        return gap_corr, speed_corr, speed_diff_corr
