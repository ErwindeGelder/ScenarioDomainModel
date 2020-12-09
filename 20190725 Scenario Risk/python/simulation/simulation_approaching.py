""" Simulation of the scenario "approaching slower vehicle".

Creation date: 2020 08 13
Author(s): Erwin de Gelder

Modifications:
2020 12 09: Adding idm_approaching_pars.
"""

import numpy as np
from .acc import ACCParameters
from .acc_hdm import ACCHDMParameters
from .hdm import HDMParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_braking import LeaderBraking, LeaderBrakingParameters
from .simulation_longitudinal import SimulationLongitudinal


INIT_POSITION_FOLLOWER = -200


def hdm_approaching_pars(**kwargs):
    """ Define the follower parameters based on the scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    steptime = 0.01
    amin = kwargs["amin"] if "amin" in kwargs else -10
    parameters = HDMParameters(model=IDMPlus(), speed_std=0.05, tau=20, rttc=0.01,
                               timestep=steptime,
                               parms_model=IDMParameters(speed=kwargs["vego"],
                                                         init_speed=kwargs["vego"],
                                                         init_position=INIT_POSITION_FOLLOWER,
                                                         timestep=steptime,
                                                         n_reaction=100,
                                                         thw=1.1,
                                                         safety_distance=2,
                                                         a_acc=1,
                                                         b_acc=1.5,
                                                         amin=amin))
    return parameters


def acc_approaching_pars(**kwargs):
    """ Define the ACC parameters in an approaching scenario.

    :return: Parameter object that can be passed via init_simulation.
    """
    amin = kwargs["amin"] if "amin" in kwargs else -10
    return ACCParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=INIT_POSITION_FOLLOWER,
                         n_reaction=0,
                         amin=amin)


def acc_hdm_approaching_pars(**kwargs):
    """ Define the parameters for the ACCHDM model.

    :return: Parameter object that can be passed via init_simulation.
    """
    return ACCHDMParameters(speed=kwargs["vego"],
                            init_speed=kwargs["vego"],
                            init_position=INIT_POSITION_FOLLOWER,
                            n_reaction=0,
                            driver_parms=hdm_approaching_pars(**kwargs))


def idm_approaching_pars(**kwargs):
    """ Define the parameters for the IDM model.

    The reaction time is sampled from the lognormal distribution mentioned in
    Wang & Stamatiadis (2014).

    :param kwargs: Parameter object that can be passed via init_simulation.
    """
    reactiontime = np.random.lognormal(np.log(.92**2/np.sqrt(.92**2+.28**2)),
                                       np.sqrt(np.log(1+.28**2/.92**2)))
    steptime = 0.01
    amin = kwargs["amin"] if "amin" in kwargs else -10
    return IDMParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=INIT_POSITION_FOLLOWER,
                         timestep=steptime,
                         n_reaction=int(reactiontime/steptime),
                         thw=1.1,
                         safety_distance=2,
                         a_acc=1,
                         b_acc=1.5,
                         amin=amin)


class SimulationApproaching(SimulationLongitudinal):
    """ Class for simulation the scenario "approaching slower vehicle". """
    def __init__(self, follower, follower_parameters, **kwargs):
        SimulationLongitudinal.__init__(self, LeaderBraking(), self._leader_parameters, follower,
                                        follower_parameters, **kwargs)

    @staticmethod
    def _leader_parameters(**kwargs):
        """ Return the paramters for the leading vehicle. """
        return LeaderBrakingParameters(init_position=0,
                                       init_speed=kwargs["vego"]*kwargs["ratio_vtar_vego"],
                                       average_deceleration=1,
                                       speed_difference=0,
                                       tconst=5)
