""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 06 08
Author(s): Erwin de Gelder

Modifications:
2020 08 06 Use correct data. Update case study.
"""

import argparse
import os
import numpy as np
from databaseemulator import DataBaseEmulator
from domain_model import Scenario
from simulation import SimulationLeadBraking
from case_study import CaseStudy, CaseStudyOptions, default_process_result


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--overwrite", help="Whether to overwrite old results", action="store_true")
ARGS = PARSER.parse_args()


def parameters_lead_braking() -> np.ndarray:
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


def check_validity_lead_braking(par: np.ndarray) -> bool:
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


if __name__ == "__main__":
    CaseStudy(CaseStudyOptions(overwrite=ARGS.overwrite,
                               filename_kde_pars="lead_braking.p",
                               filename_mc="lead_braking_mc.csv",
                               filename_is="lead_braking_is.csv",
                               filename_kde_is="lead_braking_is.p",
                               func_parameters=parameters_lead_braking,
                               func_validity_parameters=check_validity_lead_braking,
                               func_process_result=default_process_result,
                               simulator=SimulationLeadBraking(),
                               parameters=["v0", "amean", "dv"],
                               init_par_is=[20, 2, 20],
                               mcmc_step=np.array([2.0, 0.3, 2.0]),
                               nmc=10000, nis=2000))
