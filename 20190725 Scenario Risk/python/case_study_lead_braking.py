""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 06 08
Author(s): Erwin de Gelder

Modifications:
2020 08 06 Use correct data. Update case study.
2020 08 11 Update the case study. Use meta-model to get importance density.
"""

import argparse
import os
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from databaseemulator import DataBaseEmulator
from domain_model import Scenario
from simulation import SimulationLeadBraking, ACC, acc_parameters
from case_study import CaseStudy, CaseStudyOptions, default_process_result


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--overwrite", help="Whether to overwrite old results", action="store_true")
ARGS = PARSER.parse_args()


def parameters_lead_braking() -> np.ndarray:
    """ Return the parameters for lead vehicle braking.

    The parameters are:
    1. Initial speed
    2. Average deceleration
    3. Speed difference (positive means braking, i.e., vstart-vend)

    :return: The numpy array with the parameters.
    """
    data = DataBaseEmulator(os.path.join("data", "5_scenarios", "lead_braking.json"))
    pars = []
    for i in range(len(data.collections["scenario"])):
        scenario = data.get_ordered_item("scenario", i)
        vstart, vdiff, amean = 0, 0, 0
        for activity in scenario.activities:
            if activity.name == "deceleration target":
                vstart, vend = activity.get_state(time=[activity.tstart, activity.tend])[0]
                vdiff = vstart-vend
                amean = vdiff/(activity.tend-activity.tstart)
                break

        if vstart > 0 and vdiff > 0 and amean > 0:
            pars.append([vstart, amean, vdiff])
    return np.array(pars)


def par_lead_braking_alternative() -> np.ndarray:
    """ Return the parameters for lead vehicle braking.

    The parameters are slightly different from the `parameters_lead_braking`
    function. The third parameter is now the ratio of the speed difference over
    the initial speed.

    The parameters are:
    1. Initial speed
    2. Average deceleration
    3. Speed difference / initial speed

    :return: The numpy array with the parameters."""
    pars = parameters_lead_braking()
    pars[:, 2] = pars[:, 2] / pars[:, 0]
    return pars


def parameters_ego_braking(scenario: Scenario) -> np.ndarray:
    """ Get the parameters of the ego-braking scenario.

    It is assumed that the secnario contains only one scenario.
    This activity describes the braking activity of the ego
    vehicle. The following parameters are obtained:
    - Initial speed
    - Average deceleration
    - Speed difference

    :param scenario: The scenario describing the braking activity of the ego
        vehicle.
    :return: The mentioned parameters.
    """
    activity = scenario.activities[0]
    vstart_vend = activity.get_state(time=[activity.tstart, activity.tend])
    vstart = vstart_vend[0]
    vdiff = (vstart - vstart_vend[1])
    amean = vdiff / (activity.tend - activity.tstart)
    return np.array([vstart, amean, vdiff])


def check_validity_lead_braking(par: Union[List, np.ndarray]) -> bool:
    """ Check whether the parameters are valid.

    :param par: A vector of the parameters.
    :return: Whether the parameters are valid.
    """
    # Speed should be decreasing.
    if par[2] <= 0:
        return False

    # End speed should not be negative.
    if par[2] > par[0]:
        return False

    # Mean deceleration should be positive.
    if par[1] <= 0:
        return False

    return True


def check_validity_lead_braking_alt(par: Union[List, np.ndarray]) -> bool:
    """ Check whether the parameters are valid.

    In this case, it is assumed that the third parameter is NOT the speed
    difference, but the ratio of the speed difference and the initial speed.

    :param par: A vector of the parameters.
    :return: Whether the parameters are valid.
    """
    return check_validity_lead_braking([par[0], par[1], par[2]*par[0]])


if __name__ == "__main__":
    default_pars = dict(overwrite=ARGS.overwrite,
                        func_parameters=par_lead_braking_alternative,
                        func_validity_parameters=check_validity_lead_braking_alt,
                        func_process_result=default_process_result,
                        parameters=["v0", "amean", "ratio_dv_v0"],
                        init_par_is=[20, 2, 1.0],
                        mcmc_step=np.array([2.0, 0.3, 0.1]),
                        nmc=10000,
                        grid_parameters=[np.linspace(5, 60, 12),
                                         np.linspace(0.5, 6, 12),
                                         np.linspace(0.1, 1, 10)])
    print("HDM:")
    hdm_pars = dict(filename_prefix="lead_braking_hdm",
                    simulator=SimulationLeadBraking())
    hdm_pars.update(default_pars)
    cs = CaseStudy(CaseStudyOptions(**hdm_pars))

    plt.subplots(1, 1)
    plt.scatter(cs.df_mc["v0"], cs.df_mc["ratio_dv_v0"], c=cs.df_mc["kpi"])
    plt.xlabel("Initial speed [m/s]")
    plt.ylabel("Speed reduction ratio")
    plt.title("Monte Carlo, yellow = collision")
    xlim = plt.xlim()
    ylim = plt.ylim()

    plt.subplots(1, 1)
    plt.scatter(cs.df_is["v0"], cs.df_is["ratio_dv_v0"], c=cs.df_is["kpi"])
    plt.xlabel("Initial speed [m/s]")
    plt.ylabel("Speed reduction ratio")
    plt.title("Importance sampling, yellow = collision")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

    print()
    print("ACC:")
    acc_pars = dict(filename_prefix="lead_braking_acc.p",
                    simulator=SimulationLeadBraking(follower=ACC(),
                                                    follower_parameters=acc_parameters),)
    acc_pars.update(default_pars)
    CaseStudy(CaseStudyOptions(**acc_pars))
