""" Do the simulations for the case study of the scenario risk paper.

In this case, one parameter will be fixed, such that we only have two parameters
that are free to choose. In this way, it is much easier to visualize the
results.

Creation date: 2020 06 08
Author(s): Erwin de Gelder

Modifications:
2020 08 06 Use correct data. Update case study.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from simulation import SimulationLeadBraking
from stats import KDE, kde_from_file
from case_study import CaseStudy, CaseStudyOptions, default_process_result
from case_study_lead_braking import parameters_lead_braking, check_validity_lead_braking


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--overwrite", help="Whether to overwrite old results", action="store_true")
ARGS = PARSER.parse_args()
INIT_SPEED = 20  # [m/s]


def get_2d_kde(kde: KDE, overwrite: bool = False) -> KDE:
    """ Derive the 2D KDE, where the initial speed is fixed.

    :param kde: The original 3D KDE.
    :param overwrite: Whether to recompute the 2D KDE.
    :return: The 2D KDE.
    """
    filename_kde = os.path.join("data", "6_kde", "lead_braking_2d.p")
    if os.path.exists(filename_kde) and not overwrite:
        return kde_from_file(filename_kde)

    np.random.seed(0)
    samples = kde.conditional_sample(0, INIT_SPEED, 1000)
    new_kde = KDE(samples, scaling=True)
    new_kde.compute_bandwidth()
    new_kde.pickle(filename_kde)
    return new_kde


def check_validity_2d(par: np.ndarray) -> bool:
    """ Check whether the parameters are valid.

    Based on the original function, but with fixed initial speed.

    :param par: A vector of the parameters.
    :return: Whether the parameters are valid.
    """
    return check_validity_lead_braking(np.array([INIT_SPEED, par[0], par[1]]))


if __name__ == "__main__" or True:
    CS_DRIVER = CaseStudy(CaseStudyOptions(overwrite=ARGS.overwrite,
                                           filename_kde_pars="lead_braking.p",
                                           filename_mc="lead_braking_mc_2d.csv",
                                           filename_is="lead_braking_is_2d.csv",
                                           filename_kde_is="lead_braking_is_2d.p",
                                           func_parameters=parameters_lead_braking,
                                           func_kde_update=get_2d_kde,
                                           func_validity_parameters=check_validity_2d,
                                           func_process_result=default_process_result,
                                           simulator=SimulationLeadBraking(),
                                           parameters=["amean", "dv"],
                                           default_parameters=dict(v0=INIT_SPEED),
                                           init_par_is=[2, 20],
                                           mcmc_step=np.array([0.3, 2.0]),
                                           nmc=1000, nis=1000))

    # Visualize the 2D density of the parameters.
    SCALING_KDE = CS_DRIVER.kde.probability([0, 0], [400., INIT_SPEED])
    AMEAN_LIM = (0., 5.)
    DV_LIM = (0., 20.)
    AMEANS, DVS = np.meshgrid(np.linspace(AMEAN_LIM[0], AMEAN_LIM[1], 100),
                              np.linspace(DV_LIM[0], DV_LIM[1], 50))
    PDF = CS_DRIVER.kde.score_samples(np.concatenate((AMEANS[:, :, np.newaxis],
                                                      DVS[:, :, np.newaxis]), axis=2))
    FIG, AXES = plt.subplots(1, 1)
    CONTOUR = AXES.contourf(AMEANS, DVS, PDF/SCALING_KDE)
    FIG.colorbar(CONTOUR)
    AXES.set_xlabel("Mean deceleration [m/s$^2$]")
    AXES.set_ylabel("Speed difference [m/s]")
    AXES.set_title("Probability density")

    # Visualize the results of the initial simulations.
    FIG, AXES = plt.subplots(1, 1)
    SCATTER = AXES.scatter(CS_DRIVER.df_mc["amean"], CS_DRIVER.df_mc["dv"],
                           c=CS_DRIVER.df_mc["result"], cmap="seismic")
    AXES.set_title("Monte Carlo simulations results")
    FIG.colorbar(SCATTER)
    FIG, AXES = plt.subplots(1, 1)
    SCATTER = AXES.scatter(CS_DRIVER.df_mc["amean"], CS_DRIVER.df_mc["dv"],
                           c=CS_DRIVER.df_mc["kpi"], cmap="seismic")
    AXES.set_title("Monte Carlo simulations results")
    FIG.colorbar(SCATTER)

    # Visualize the 2D importance density.
    FIG, AXES = plt.subplots(1, 1)
    SCALING_KDE = CS_DRIVER.kde_is.probability([0., 0.], [400., INIT_SPEED])
    PDF = CS_DRIVER.kde_is.score_samples(np.concatenate((AMEANS[:, :, np.newaxis],
                                                        DVS[:, :, np.newaxis]), axis=2))
    AXES.contourf(AMEANS, DVS, PDF/SCALING_KDE)
    AXES.set_xlabel("Mean deceleration [m/s$^2$]")
    AXES.set_ylabel("Speed difference [m/s]")
    AXES.set_title("Importance density")

    plt.show()
