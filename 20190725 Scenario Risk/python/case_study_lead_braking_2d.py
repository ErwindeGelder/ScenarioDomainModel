""" Do the simulations for the case study of the scenario risk paper.

In this case, one parameter will be fixed, such that we only have two parameters
that are free to choose. In this way, it is much easier to visualize the
results.

Creation date: 2020 06 08
Author(s): Erwin de Gelder

Modifications:
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from simulation import SimulationLeadBraking
from stats import KDE
from case_study import CaseStudy, CaseStudyOptions, kde_from_json, kde_to_json
from case_study_lead_braking import parameters_ego_braking, check_validity_lead_braking


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
    filename_kde = os.path.join("data", "6_kde", "ego_braking_2d.json")
    if os.path.exists(filename_kde) and not overwrite:
        return kde_from_json(filename_kde)

    np.random.seed(0)
    ncdf = 100
    nkde = 1000

    # Get the CDF of the original KDE for varying amean and fixed INIT_SPEED.
    xcdf = np.linspace((np.min(kde.data[:, 1])-3*kde.bandwidth)*kde.data_helpers.std[1],
                       (np.max(kde.data[:, 1])+3*kde.bandwidth)*kde.data_helpers.std[1],
                       ncdf)
    xone = np.ones(ncdf)
    ycdf = (kde.cdf(np.vstack((xone*(INIT_SPEED+1e-4), xcdf, xone*1000)).T)-
            kde.cdf(np.vstack((xone*INIT_SPEED, xcdf, xone*1000)).T))/1e-4

    data = np.zeros((nkde, 2))
    xcdf2 = np.linspace((np.min(kde.data[:, 2])-3*kde.bandwidth)*kde.data_helpers.std[2],
                        (np.max(kde.data[:, 2])+3*kde.bandwidth)*kde.data_helpers.std[2],
                        ncdf)
    print("Obtaining the 2D KDE")
    for i in tqdm(range(nkde)):
        data[i, 0] = np.interp(np.random.rand()*ycdf[-1], ycdf, xcdf)
        ycdf2 = ((kde.cdf(np.vstack((xone*(INIT_SPEED+1e-4), xone*(data[i, 0]+1e-4), xcdf2)).T) -
                  kde.cdf(np.vstack((xone*(INIT_SPEED+1e-4), xone*(data[i, 0]-1e-4), xcdf2)).T) -
                  kde.cdf(np.vstack((xone*(INIT_SPEED-1e-4), xone*(data[i, 0]+1e-4), xcdf2)).T) +
                  kde.cdf(np.vstack((xone*(INIT_SPEED-1e-4), xone*(data[i, 0]-1e-4), xcdf2)).T)) /
                 (4*1e-8))
        data[i, 1] = np.interp(np.random.rand()*ycdf2[-1], ycdf2, xcdf2)

    kde = KDE(data, scaling=True)
    kde.compute_bandwidth()
    print("2D KDE bandwidth: {:.3f}".format(kde.bandwidth))

    # Write the KDE through a JSON file.
    kde_to_json(filename_kde, kde)
    return kde


def check_validity_2d(par: np.ndarray) -> bool:
    """ Check whether the parameters are valid.

    Based on the original function, but with fixed initial speed.

    :param par: A vector of the parameters.
    :return: Whether the parameters are valid.
    """
    return check_validity_lead_braking(np.array([INIT_SPEED, par[0], par[1]]))


if __name__ == "__main__" or True:
    CASE_STUDY = CaseStudy(CaseStudyOptions(overwrite=ARGS.overwrite,
                                            filename_data="ego_braking.json",
                                            filename_kde_pars="ego_braking.json",
                                            filename_kde_mcmc="ego_braking_mcmc_2d.json",
                                            filename_df="lead_braking_2d.csv",
                                            filename_mc="lead_braking_mc_2d.csv",
                                            filename_dfis="lead_braking_is_2d.csv",
                                            func_parameters=parameters_ego_braking,
                                            func_validity_parameters=check_validity_2d,
                                            func_kde_update=get_2d_kde,
                                            simulator=SimulationLeadBraking(),
                                            parameters=["amean", "dv"],
                                            grid_parameters=[np.linspace(0.5, 5, 33),
                                                             np.linspace(5, INIT_SPEED, 17)],
                                            default_parameters=dict(v0=INIT_SPEED),
                                            init_par_mcmc=[3., 10.],
                                            mcmc_step=np.array([0.5, 2.]),
                                            nmc=1000,
                                            nthinning=100,
                                            nburnin=100,
                                            nmcmc=1000,
                                            nsimis=1000))

    # Visualize the 2D density of the parameters.
    SCALING_KDE = (CASE_STUDY.kde.cdf(np.array([[400., INIT_SPEED]]))[0] -
                   CASE_STUDY.kde.cdf(np.array([[400., 0.]]))[0] -
                   CASE_STUDY.kde.cdf(np.array([[0., INIT_SPEED]]))[0] +
                   CASE_STUDY.kde.cdf(np.array([[0., 0.]]))[0])
    AMEAN_LIM = (0., 3.)
    DV_LIM = (0., 20.)
    AMEANS, DVS = np.meshgrid(np.linspace(AMEAN_LIM[0], AMEAN_LIM[1], 100),
                              np.linspace(DV_LIM[0], DV_LIM[1], 50))
    PDF = CASE_STUDY.kde.score_samples(np.concatenate((AMEANS[:, :, np.newaxis],
                                                       DVS[:, :, np.newaxis]), axis=2))
    FIG, AXES = plt.subplots(1, 1)
    CONTOUR = AXES.contourf(AMEANS, DVS, PDF/SCALING_KDE)
    FIG.colorbar(CONTOUR)
    AXES.set_xlabel("Mean deceleration [m/s$^2$]")
    AXES.set_ylabel("Speed difference [m/s]")
    AXES.set_title("Probability density")

    # Visualize the results of the initial simulations.
    FIG, AXES = plt.subplots(1, 1)
    SCATTER = AXES.scatter(CASE_STUDY.df["amean"], CASE_STUDY.df["dv"], c=CASE_STUDY.df["result"],
                           cmap="seismic")
    FIG.colorbar(SCATTER)

    # Visualize the 2D importance density.
    SCALING_KDE = (CASE_STUDY.kde_is.cdf(np.array([[400., INIT_SPEED]]))[0] -
                   CASE_STUDY.kde_is.cdf(np.array([[400., 0.]]))[0] -
                   CASE_STUDY.kde_is.cdf(np.array([[0., INIT_SPEED]]))[0] +
                   CASE_STUDY.kde_is.cdf(np.array([[0., 0.]]))[0])
    AMEANS, DVS = np.meshgrid(np.linspace(AMEAN_LIM[0], AMEAN_LIM[1], 100),
                              np.linspace(DV_LIM[0], DV_LIM[1], 50))
    PDF = CASE_STUDY.kde_is.score_samples(np.concatenate((AMEANS[:, :, np.newaxis],
                                                          DVS[:, :, np.newaxis]), axis=2))
    FIG, AXES = plt.subplots(1, 1)
    CONTOUR = AXES.contourf(AMEANS, DVS, PDF/SCALING_KDE)
    FIG.colorbar(CONTOUR)
    AXES.set_xlabel("Mean deceleration [m/s$^2$]")
    AXES.set_ylabel("Speed difference [m/s]")
    AXES.set_title("Importance density")

    plt.show()
