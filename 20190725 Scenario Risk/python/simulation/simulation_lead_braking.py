""" Simulation of the scenario "lead vehicle braking".

Creation date: 2020 05 29
Author(s): Erwin de Gelder

Modifications:
2020 06 22 Parameters based on Treiber et al. (2006).
2020 06 23 Add possibility to use any driver model for follower.
2020 08 11 Allow to use option 'ratio_dv_v0' instead of 'dv'.
2020 08 13 Use SimulationLongitudinal as superclass to do the actual simulation.
"""

from .acc import ACC, ACCParameters
from .acc_hdm import ACCHDM, ACCHDMParameters
from .cacc import CACC, CACCParameters
from .eidm import EIDMParameters
from .hdm import HDM, HDMParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_braking import LeaderBraking, LeaderBrakingParameters
from .simulation_longitudinal import SimulationLongitudinal


def hdm_lead_braking_pars(**kwargs):
    """ Define the follower parameters based on the scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    steptime = 0.01
    safety_distance = 2.0
    thw = 1.1
    init_distance = safety_distance + init_speed * thw
    parameters = HDMParameters(model=IDMPlus(), speed_std=0.05, tau=20, rttc=0.01,
                               timestep=steptime,
                               parms_model=IDMParameters(speed=init_speed*1.2,
                                                         init_speed=init_speed,
                                                         init_position=-init_distance,
                                                         timestep=0.01,
                                                         n_reaction=100,
                                                         thw=1.1,
                                                         safety_distance=2,
                                                         a_acc=1,
                                                         b_acc=1.5))
    return parameters


def eidm_lead_braking_pars(**kwargs):
    """ Define the follower parameters based on the scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    safety_distance = 2.0
    thw = 1.1
    init_distance = safety_distance + init_speed * thw
    parameters = EIDMParameters(speed=init_speed*1.2,
                                init_speed=init_speed,
                                init_position=-init_distance,
                                timestep=0.01,
                                n_reaction=0,
                                thw=1.1,
                                safety_distance=2,
                                amin=-8,
                                a_acc=1,
                                b_acc=1.5,
                                coolness=0.99)
    return parameters


def acc_lead_braking_pars(**kwargs):
    """ Define the ACC parameters of the follower based on scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    safety_distance = ACC.safety_distance(init_speed)
    default_parameters = ACCParameters()
    thw = default_parameters.thw
    init_distance = safety_distance + init_speed * thw
    parameters = ACCParameters(speed=init_speed,
                               init_speed=init_speed,
                               init_position=-init_distance,
                               n_reaction=0)
    return parameters


def acc_hdm_lead_braking_pars(**kwargs):
    """ Define the parameters for the ACCHDM model.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    safety_distance = ACCHDM.safety_distance(init_speed)
    default_parameters = ACCHDMParameters()
    thw = default_parameters.thw
    init_distance = safety_distance + init_speed * thw
    parameters = ACCHDMParameters(speed=init_speed,
                                  init_speed=init_speed,
                                  init_position=-init_distance,
                                  n_reaction=0,
                                  driver_parms=hdm_lead_braking_pars(**kwargs))
    return parameters


def cacc_lead_braking_pars(**kwargs):
    """ Define the CACC parameters of the follower based on scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    safety_distance = CACC.safety_distance(init_speed)
    default_parameters = CACCParameters()
    thw = default_parameters.thw
    init_distance = safety_distance + init_speed * thw
    parameters = CACCParameters(speed=init_speed,
                                init_speed=init_speed,
                                init_position=-init_distance,
                                n_reaction=0)
    return parameters


class SimulationLeadBraking(SimulationLongitudinal):
    """ Class for simulation the scenario "lead vehicle braking".

    Attributes:
        leader(LeaderBraking)
        follower - any given driver model (by default, HDM is used)
        follower_parameters - function for obtaining the parameters.
    """
    def __init__(self, follower=None, follower_parameters=None, **kwargs):
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
