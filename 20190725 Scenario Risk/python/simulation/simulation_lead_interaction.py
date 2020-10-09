""" Simulation the situation in which the host vehicle interacts with its lead vehicle.

Creation date: 2020 06 18
Author(s): Erwin de Gelder

Modifications:
2020 10 08 Use LeaderInteraction to model the leader vehicle.
"""

import matplotlib.pyplot as plt
import numpy as np
from .hdm import HDM, HDMParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_interaction import LeaderInteraction, LeaderInteractionParameters
from .simulation_longitudinal import SimulationLongitudinal


class SimulationLeadInteraction(SimulationLongitudinal):
    """ Class for simulation the scenario "lead vehicle interaction".

    Attributes:
        leader(LeaderBraking)
        follower - any given driver model (by default, HDM is used)
        follower_parameters - function for obtaining the parameters.
    """
    def __init__(self, follower, follower_parameters, **kwargs):
        # Instantiate the vehicles.
        if follower is None:
            follower = HDM()
        if follower_parameters is None:
            follower_parameters = hdm_lead_braking_pars
        SimulationLongitudinal.__init__(self, LeaderBraking(), self._leader_parameters, follower,
                                        follower_parameters, **kwargs)

    @staticmethod
    def _leader_parameters(**kwargs):
        """ Return the paramters for the leading vehicle. """
        return LeaderBrakingParameters(init_position=0,
                                       init_speed=kwargs["v0"],
                                       average_deceleration=kwargs["amean"],
                                       speed_difference=kwargs["dv"],
                                       tconst=5)

    def init_simulation(self, **kwargs) -> None:
        """ Initialize the simulation.

        :param kwargs: The parameters: (v0, amean, dv) OR (v0, amean, ratio_dv_v0).
        """
        if "dv" not in kwargs:
            kwargs["dv"] = kwargs["v0"] * kwargs["ratio_dv_v0"]

        SimulationLongitudinal.init_simulation(self, **kwargs)
