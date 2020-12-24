""" Simulations for the case study of the scenario risk paper. Scenario: approaching slower vehicle.

Creation date: 2020 08 13
Author(s): Erwin de Gelder

Modifications:
"""

import argparse
import os
from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from domain_model import StateVariable, DocumentManagement
from simulation import SimulationApproaching, IDMPlus, ACC, idm_approaching_pars, \
    acc_approaching_pars
from case_study import CaseStudy, CaseStudyOptions, default_process_result


def parameters_approaching() -> np.ndarray:
    """ Return the parameters for lead vehicle braking.

    The parameters are:
    1. Ego speed;
    2. Ratio target speed and ego speed [0-1); meaning that target is always slower.

    :return: The numpy array with the parameters.
    """
    data = DocumentManagement(os.path.join("data", "5_scenarios", "approaching_vehicle2.json"))
    pars = []
    for key in data.collections["scenario"]:
        scenario = data.get_item("scenario", key)
        vego = scenario.get_state(scenario.get_actor_by_name("ego vehicle"), StateVariable.SPEED,
                                  scenario.get_tstart())
        vtarget = scenario.get_state(scenario.get_actor_by_name("target vehicle"),
                                     StateVariable.LON_TARGET, scenario.get_tstart())[0]
        if vego > vtarget:
            pars.append([vego, vtarget/vego])

    return np.array(pars)


def check_validity_approaching(par: Union[List, np.ndarray]) -> bool:
    """ Check whether the parameters are valid.

    :param par: A vector of the parameters.
    :return: Whether the parameters are valid.
    """
    # Ego speed must be strictly positive.
    if par[0] <= 0:
        return False

    # Relatiev speed of target speed must not be negative and less than 1.
    if par[1] < 0 or par[1] >= 1:
        return False

    return True


def plot_result(case_study: CaseStudy, title: str = "") -> Figure:
    """ Plot some results. """

    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    ax1.scatter(case_study.df_mc["vego"], case_study.df_mc["ratio_vtar_vego"],
                c=case_study.df_mc["kpi"])
    ax1.set_xlabel("Speed ego vehicle [m/s]")
    ax1.set_ylabel("Ratio target/ego speed")
    ax1.set_title("{:s}\nMonte Carlo, yellow = collision".format(title))
    # xlim = ax1.get_xlim()
    # ylim = ax1.get_ylim()
    #
    # ax2.scatter(case_study.df_is["vego"], case_study.df_is["ratio_vtar_vego"],
    #             c=case_study.df_is["kpi"])
    # ax2.set_xlabel("Speed ego vehicle [m/s]")
    # ax2.set_ylabel("Ratio target/ego speed")
    # ax2.set_title("{:s}\nImportance sampling, yellow = collision".format(title))
    # ax2.set_xlim(xlim)
    # ax2.set_ylim(ylim)
    #
    # values, indices = np.unique(case_study.df_grid[['vego', 'ratio_vtar_vego']].values, axis=0,
    #                             return_inverse=True)
    # collisions = np.zeros(len(values))
    # for i in range(len(values)):
    #     collisions[i] = np.sum(case_study.df_grid.loc[indices == i, "kpi"])
    # ax3.scatter(values[:, 0], values[:, 1], c=collisions)
    # ax3.set_xlabel("Speed ego vehicle [m/s]")
    # ax3.set_ylabel("Ratio target/ego speed")
    # ax3.set_title("{:s}\nyellower = more collision (total={:.0f})".format(title, sum(collisions)))
    return figure


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--overwrite", help="Whether to overwrite old results", action="store_true")
    ARGS = PARSER.parse_args()

    DEFAULT_PARS = dict(overwrite=ARGS.overwrite,
                        func_parameters=parameters_approaching,
                        func_validity_parameters=check_validity_approaching,
                        func_process_result=default_process_result,
                        parameters=["vego", "ratio_vtar_vego"],
                        default_parameters=dict(amin=-3),
                        do_importance_sampling_mcmc=False,
                        do_importance_sampling_direct=False,
                        init_par_is=[45, 0],
                        mcmc_step=np.array([4.0, 0.1]),
                        nmc=1000,
                        grid_parameters=[np.linspace(10, 50, 17),
                                         np.linspace(0.1, 1, 19)])

    print("IDM+:")
    IDM_PARS = dict(filename_prefix="approaching_idmplus",
                    simulator=SimulationApproaching(follower=IDMPlus(),
                                                    follower_parameters=idm_approaching_pars))
    IDM_PARS.update(DEFAULT_PARS)
    CASE_STUDY = CaseStudy(CaseStudyOptions(**IDM_PARS))
    plot_result(CASE_STUDY, "IDM").savefig(os.path.join("figs", "approaching_idmplus.png"))

    print()
    print("ACC:")
    ACC_PARS = dict(filename_prefix="approaching_acc",
                    simulator=SimulationApproaching(follower=ACC(),
                                                    follower_parameters=acc_approaching_pars))
    ACC_PARS.update(DEFAULT_PARS)
    CASE_STUDY = CaseStudy(CaseStudyOptions(**ACC_PARS))
    plot_result(CASE_STUDY, "ACC").savefig(os.path.join("figs", "approaching_acc.png"))

    # print()
    # print("ACC, FCW, and HDM:")
    # ACC_HDM_PARS = dict(filename_prefix="approaching_acchdm",
    #                     simulator=SimulationApproaching(
    #                         follower=ACCHDM(), follower_parameters=acc_hdm_approaching_pars))
    # ACC_HDM_PARS.update(DEFAULT_PARS)
    # CASE_STUDY = CaseStudy(CaseStudyOptions(**ACC_HDM_PARS))
    # plot_result(CASE_STUDY, "ACC, FCW, and HDM").savefig(os.path.join("figs", "approaching_acchdm.png"))
    plt.show()
