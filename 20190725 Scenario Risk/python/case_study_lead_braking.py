""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 06 08
Author(s): Erwin de Gelder

Modifications:
2020 08 06 Use correct data. Update case study.
2020 08 11 Update the case study. Use meta-model to get importance density.
2020 08 12 Provide plot functionality for each case study.
"""

import argparse
import os
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from domain_model import Scenario
from simulation import SimulationLeadBraking, ACC, acc_lead_braking_pars, ACCHDM, \
    acc_hdm_lead_braking_pars
from case_study import CaseStudy, CaseStudyOptions, default_process_result


# def parameters_lead_braking() -> np.ndarray:
#     """ Return the parameters for lead vehicle braking.
#
#     The parameters are:
#     1. Initial speed
#     2. Average deceleration
#     3. Speed difference (positive means braking, i.e., vstart-vend)
#
#     :return: The numpy array with the parameters.
#     """
#     data = DataBaseEmulator(os.path.join("data", "5_scenarios", "lead_braking.json"))
#     pars = []
#     for i in range(len(data.collections["scenario"])):
#         scenario = data.get_ordered_item("scenario", i)
#         vstart, vdiff, amean = 0, 0, 0
#         for activity in scenario.activities:
#             if activity.name == "deceleration target":
#                 vstart, vend = activity.get_state(time=[activity.tstart, activity.tend])[0]
#                 vdiff = vstart-vend
#                 amean = vdiff/(activity.tend-activity.tstart)
#                 break
#
#         if vstart > 0 and vdiff > 0 and amean > 0:
#             pars.append([vstart, amean, vdiff])
#     return np.array(pars)


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


def plot_result(case_study: CaseStudy, title: str = ""):
    """ Plot some results. """

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    ax1.scatter(case_study.df_mc["v0"], case_study.df_mc["ratio_dv_v0"], c=case_study.df_mc["kpi"])
    ax1.set_xlabel("Initial speed [m/s]")
    ax1.set_ylabel("Speed reduction ratio")
    ax1.set_title("{:s}\nMonte Carlo, yellow = collision".format(title))
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    ax2.scatter(case_study.df_is["v0"], case_study.df_is["ratio_dv_v0"], c=case_study.df_is["kpi"])
    ax2.set_xlabel("Initial speed [m/s]")
    ax2.set_ylabel("Speed reduction ratio")
    ax2.set_title("{:s}\nImportance sampling, yellow = collision".format(title))
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    values, indices = np.unique(case_study.df_grid[['v0', 'ratio_dv_v0']].values, axis=0,
                                return_inverse=True)
    collisions = np.zeros(len(values))
    for i in range(len(values)):
        collisions[i] = np.sum(case_study.df_grid.loc[indices == i, "kpi"])
    ax3.scatter(values[:, 0], values[:, 1], c=collisions)
    ax3.set_xlabel("Initial speed [m/s]")
    ax3.set_ylabel("Speed reduction ratio")
    ax3.set_title("{:s}\nyellower = more collision (total={:.0f})".format(title, sum(collisions)))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--overwrite", help="Whether to overwrite old results", action="store_true")
    ARGS = PARSER.parse_args()

    DEFAULT_PARS = dict(overwrite=ARGS.overwrite,
                        func_parameters=par_lead_braking_alternative,
                        func_validity_parameters=check_validity_lead_braking_alt,
                        func_process_result=default_process_result,
                        parameters=["v0", "amean", "ratio_dv_v0"],
                        init_par_is=[20, 6, 1.0],
                        mcmc_step=np.array([2.0, 0.3, 0.1]),
                        nmc=1000,
                        grid_parameters=[np.linspace(5, 60, 12),
                                         np.linspace(0.5, 6, 12),
                                         np.linspace(0.1, 1, 10)])
    print("HDM:")
    HDM_PARS = dict(filename_prefix="lead_braking_hdm",
                    simulator=SimulationLeadBraking())
    HDM_PARS.update(DEFAULT_PARS)
    CASE_STUDY = CaseStudy(CaseStudyOptions(**HDM_PARS))
    plot_result(CASE_STUDY, "HDM")

    print()
    print("ACC:")
    ACC_PARS = dict(filename_prefix="lead_braking_acc",
                    simulator=SimulationLeadBraking(follower=ACC(),
                                                    follower_parameters=acc_lead_braking_pars))
    ACC_PARS.update(DEFAULT_PARS)
    CASE_STUDY = CaseStudy(CaseStudyOptions(**ACC_PARS))
    plot_result(CASE_STUDY, "ACC")

    print()
    print("ACC, FCW, and HDM:")
    ACC_HDM_PARS = dict(filename_prefix="lead_braking_acchdm",
                        simulator=SimulationLeadBraking(
                            follower=ACCHDM(), follower_parameters=acc_hdm_lead_braking_pars))
    ACC_HDM_PARS.update(DEFAULT_PARS)
    CASE_STUDY = CaseStudy(CaseStudyOptions(**ACC_HDM_PARS))
    plot_result(CASE_STUDY, "ACC, FCW, and HDM")
    plt.show()
