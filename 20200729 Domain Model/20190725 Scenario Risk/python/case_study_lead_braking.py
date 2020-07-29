""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 06 08
Author(s): Erwin de Gelder

Modifications:
"""

import numpy as np
from domain_model import Scenario
from simulation import SimulationLeadBraking
from case_study import CaseStudy, CaseStudyOptions


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
    CaseStudy(CaseStudyOptions(overwrite=False,
                               filename_data="ego_braking.json",
                               filename_kde_pars="ego_braking.json",
                               filename_kde_mcmc="ego_braking_mcmc.json",
                               filename_df="lead_braking.csv",
                               filename_dfis="lead_braking_is.csv",
                               func_parameters=parameters_ego_braking,
                               func_validity_parameters=check_validity_lead_braking,
                               simulator=SimulationLeadBraking(),
                               parameter_columns=["v0", "amean", "dv"],
                               init_par_mcmc=[10., 3., 10.],
                               mcmc_step=np.array([2., 0.5, 2.]),
                               nsim=1000,
                               nthinning=100,
                               nburnin=100,
                               nmcmc=1000,
                               nsimis=1000))
