""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
"""

import os
from typing import Callable, List, Union
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from domain_model import Scenario
from simulation import SimulationLeadBraking, Simulator
from stats import KDE
from databaseemulator import DataBaseEmulator
from options import Options


OVERWRITE = False  # If true, all results will be obtained again.


class CaseStudyOptions(Options):
    """ Options for a case study. """
    filename_data: str = ""
    filename_kde_pars: str = ""
    filename_kde_mcmc: str = ""
    filename_df: str = ""

    func_parameters: Callable = None
    func_validity_parameters: Callable = None

    simulator: Simulator = None

    parameter_columns: List[str] = []
    init_par_mcmc: List[float] = []

    mcmc_step: Union[float, np.ndarray] = 0.5

    nsim: int = 1000
    nmcmc: int = 1000
    nburnin: int = 100
    nthinning: int = 10


def kde_to_json(filename: str, kde: KDE) -> None:
    """ Write a KDE to a json file.

    :param filename: The full name of the location of the KDE.
    :param kde: The KDE object.
    """
    save = dict(data=kde.data.tolist(),
                std=kde.data_helpers.std.tolist(),
                bandwidth=kde.bandwidth)
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    with open(filename, "w") as file:
        json.dump(save, file)


def kde_from_json(filename: str) -> KDE:
    """ Obtain a KDE from a json file.

    :param filename: The full name of the location of the KDE.
    :return: The KDE object.
    """
    with open(filename, "r") as file:
        save = json.load(file)
    kde = KDE(np.array(save["data"]), scaling=True)
    kde.data_helpers.std = np.array(save["std"])  # Data is already scaled, so provide std.
    kde.bandwidth = save["bandwidth"]
    return kde


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


def index_closest(pars_orig: np.ndarray, par: np.ndarray) -> int:
    """ Return the index of the closest parameters vector.

    :param pars_orig: The original parameters.
    :param par: The new parameter vector.
    :return: The index of the closest original parameter vector.
    """
    return np.sum((pars_orig - par)**2, axis=1).argmin()


class CaseStudy:
    """ Object for performing a case study. """
    def __init__(self, options: CaseStudyOptions):
        self.options = options
        self.kde = self.get_kde()
        self.df = self.initial_simulation()
        self.mcmc_simulations()

    def get_kde(self) -> KDE:
        """ Return the KDE for the braking parameters.

        For now, the scenarios with the ego vehicle braking are used. This should
        later be changed to scenarios with the lead vehicle braking.

        :return: KDE of the braking parameters.
        """
        filename_kde = os.path.join("data", "6_kde", self.options.filename_kde_pars)
        if os.path.exists(filename_kde) and not OVERWRITE:
            kde = kde_from_json(filename_kde)
        else:
            # Create the KDE.
            filename_data = os.path.join("data", "5_scenarios", "ego_braking.json")
            dbe = DataBaseEmulator(filename_data)
            nscenarios = len(dbe.collections["scenario"])
            pars = np.array([self.options.func_parameters(dbe.get_item("scenario", i))
                             for i in range(nscenarios)])
            kde = KDE(pars, scaling=True)
            kde.compute_bandwidth()
            print("KDE bandwidth ego braking: {:.3f}".format(kde.bandwidth))

            # Write the KDE through a JSON file.
            kde_to_json(filename_kde)

        return kde

    def initial_simulation(self) -> pd.DataFrame:
        """ Do the initial simulations.

        :return: The resulting probabilities of failures (P(B)).
        """
        filename_df = os.path.join("data", "7_simulation_results", self.options.filename_df)

        if os.path.exists(filename_df) and not OVERWRITE:
            df = pd.read_csv(filename_df, index_col=0)
        else:
            # Generate the parameters.
            np.random.seed(0)
            pars = np.zeros((self.options.nsim, 3))
            i = 0
            while i < self.options.nsim:
                pars[i, :] = self.kde.sample()
                if self.options.func_validity_parameters(pars[i, :]):
                    i += 1
            df = pd.DataFrame(data=pars, columns=self.options.parameter_columns)

            # Loop through each parameter vector and obtain the simulation result.
            result = np.zeros(self.options.nsim)
            for i, par in enumerate(tqdm(pars)):
                result[i] = self.options.simulator.get_probability(tuple(par))
            df["result"] = result

            # Write to file.
            # Write the KDE through a JSON file.
            if not os.path.exists(os.path.dirname(filename_df)):
                os.mkdir(os.path.dirname(filename_df))
            df.to_csv(filename_df)

        return df

    def prob_is(self, pars: np.ndarray, par: np.ndarray) -> float:
        """ Compute the scaled importance density of the given parameters

        :param pars: The original set of parameters.
        :param par: The vector of the parameters.
        :return: The scaled importance density.
        """
        i = index_closest(pars, par)
        result = self.df.at[i, "result"]
        if result == 0.0:
            return 0
        return result*self.kde.score_samples(np.array([par]))[0]

    def mcmc_simulations(self) -> KDE:
        """ Perform the Markov Chain Monte Carlo simulation.

        :return: The resulting KDE of the importance density.
        """
        # Check if initial point is a valid scenario.
        par = np.array(self.options.init_par_mcmc)
        if not self.options.func_validity_parameters(par):
            raise ValueError("Initial parameters for MCMC are not valid.")

        # Create the list of the parameters for the importance density.
        init_pars = self.df[self.options.parameter_columns].values / self.kde.data_helpers.std
        pars = np.zeros((self.options.nmcmc, len(self.options.parameter_columns)))
        is_current = self.prob_is(init_pars, par/self.kde.data_helpers.std)
        for i in tqdm(range(-self.options.nburnin,
                            (self.options.nmcmc-1)*self.options.nthinning+1)):
            while True:
                candidate = par + (np.random.randn(len(self.options.parameter_columns)) *
                                   self.options.mcmc_step * self.kde.data_helpers.std)
                if self.options.func_validity_parameters(candidate):
                    break
            is_candidate = self.prob_is(init_pars, candidate/self.kde.data_helpers.std)
            if is_candidate > 0.0 and np.random.rand()*is_current < is_candidate:
                par = candidate.copy()
                is_current = is_candidate
            if i >= 0 and i % self.options.nthinning == 0:
                pars[i // self.options.nthinning, :] = par


# if __name__ == "__main__":
CaseStudy(CaseStudyOptions(filename_data="ego_braking.json",
                           filename_kde_pars="ego_braking.json",
                           filename_kde_mcmc="ego_braking_mcmc.json",
                           filename_df="lead_braking.csv",
                           func_parameters=parameters_ego_braking,
                           func_validity_parameters=check_validity_lead_braking,
                           simulator=SimulationLeadBraking(),
                           parameter_columns=["v0", "amean", "dv"],
                           init_par_mcmc=[10., 3., 10.],
                           mcmc_step=np.array([2., 0.5, 2.]),
                           nsim=1000,
                           nthinning=100,
                           nburnin=100,
                           nmcmc=1000))
