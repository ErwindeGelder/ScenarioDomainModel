""" Simulation of the scenario "approaching slower vehicle".

Creation date: 2020 08 13
Author(s): Erwin de Gelder

Modifications:
"""

import numpy as np
from .acc import ACCParameters
from .acc_hdm import ACCHDMParameters
from .hdm import HDMParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_braking import LeaderBraking, LeaderBrakingParameters
from .simulation_string import SimulationString


def hdm_cutin_pars(**kwargs):
    """ Define the follower parameters based on the scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    steptime = 0.01
    parameters = HDMParameters(model=IDMPlus(), speed_std=0.05, tau=20, rttc=0.01,
                               timestep=steptime,
                               parms_model=IDMParameters(speed=kwargs["vego"],
                                                         init_speed=kwargs["vego"],
                                                         init_position=-kwargs["dinit"],
                                                         timestep=steptime,
                                                         n_reaction=100,
                                                         thw=1.1,
                                                         safety_distance=2,
                                                         a_acc=1,
                                                         b_acc=1.5))
    return parameters


def idm_cutin_pars(**kwargs):
    """ Define the parameters for the IDM model.

    The reaction time is sampled from the lognormal distribution mentioned in
    Wang & Stamatiadis (2014) if it not provided through kwargs.

    :param kwargs: Parameter object that can be passed via init_simulation.
    """
    if "reactiontime" in kwargs:
        reactiontime = kwargs["reactiontime"]
    else:
        reactiontime = np.random.lognormal(np.log(.92**2/np.sqrt(.92**2+.28**2)),
                                           np.sqrt(np.log(1+.28**2/.92**2)))
    steptime = 0.01
    parms = dict()
    for parm in ["amin", "max_view"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    thw = kwargs["thw"] if "thw" in kwargs else 1.1
    return IDMParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=-kwargs["dinit"],
                         timestep=steptime,
                         n_reaction=int(reactiontime/steptime),
                         thw=thw,
                         a_acc=1,
                         b_acc=1.5,
                         **parms)


def acc_cutin_pars(**kwargs):
    """ Define the ACC parameters in an approaching scenario.

    :return: Parameter object that can be passed via init_simulation.
    """
    parms = dict()
    for parm in ["amin", "sensor_range"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    return ACCParameters(speed=kwargs["vego"],
                         init_speed=kwargs["vego"],
                         init_position=-kwargs["dinit"],
                         n_reaction=0,
                         **parms)


def acc_idm_cutin_pars(**kwargs):
    """ Define the parameters for the ACCIDM model.

    :return: Parameter object that can be passed via init_simulation.
    """
    if "amin" in kwargs:
        amin = kwargs["amin"]
    else:
        amin = -10
        kwargs["amin"] = amin
    parms = dict()
    for parm in ["k1_acc", "k2_acc"]:
        if parm in kwargs:
            parms[parm] = kwargs[parm]
    if "reactiontime" not in kwargs:
        kwargs["reactiontime"] = np.random.lognormal(np.log(.92**2/np.sqrt(.92**2+.28**2)),
                                                     np.sqrt(np.log(1+.28**2/.92**2)))
    fcw_delay = kwargs["reactiontime"]
    return ACCHDMParameters(speed=kwargs["vego"],
                            init_speed=kwargs["vego"],
                            init_position=-kwargs["dinit"],
                            n_reaction=0,
                            amin=amin,
                            driver_parms=idm_cutin_pars(**kwargs),
                            driver_model=IDMPlus(),
                            fcw_delay=fcw_delay,
                            **parms)


def acc_hdm_cutin_pars(**kwargs):
    """ Define the parameters for the ACCHDM model.

    :return: Parameter object that can be passed via init_simulation.
    """
    return ACCHDMParameters(speed=kwargs["vego"],
                            init_speed=kwargs["vego"],
                            init_position=-kwargs["dinit"],
                            n_reaction=0,
                            driver_parms=hdm_cutin_pars(**kwargs))


class SimulationCutIn(SimulationString):
    """ Class for simulation the scenario "approaching slower vehicle". """
    def __init__(self, follower, follower_parameters, **kwargs):
        SimulationString.__init__(self, [LeaderBraking(), follower],
                                  [self._leader_parameters, follower_parameters], **kwargs)
        self.min_simulation_time = 1

    @staticmethod
    def _leader_parameters(**kwargs):
        """ Return the paramters for the leading vehicle. """
        return LeaderBrakingParameters(init_position=0,
                                       init_speed=kwargs["vlead"],
                                       average_deceleration=1,
                                       speed_difference=0,
                                       tconst=5)
