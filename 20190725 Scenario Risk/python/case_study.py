""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
2020 06 08 Perform the case studies in a separate file.
2020 06 16 Do the initial simulations on a grid instead of via Monte Carlo.
"""

import itertools
import os
from typing import Callable, List, Union
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from simulation import Simulator
from stats import KDE
from databaseemulator import DataBaseEmulator
from options import Options


class CaseStudyOptions(Options):
    """ Options for a case study. """
    overwrite: bool = False

    filename_data: str = ""
    filename_kde_pars: str = ""
    filename_kde_mcmc: str = ""
    filename_df: str = ""
    filename_mc: str = ""
    filename_dfis: str = ""

    func_parameters: Callable = None
    func_kde_update = None
    func_validity_parameters: Callable = None

    simulator: Simulator = None

    parameters: List[str] = []
    grid_parameters: List[np.ndarray] = []
    init_par_mcmc: List[float] = []
    default_parameters: dict = dict()

    mcmc_step: Union[float, np.ndarray] = 0.5

    seed: int = 0
    nmc: int = 1000
    nmcmc: int = 1000
    nburnin: int = 100
    nthinning: int = 100
    nsimis: int = 1000


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
        if self.options.func_kde_update is not None:
            self.kde = self.options.func_kde_update(self.kde, overwrite=self.options.overwrite)
        self.df = self.initial_simulation()
        self.df_mc = self.initial_mc_simulation()
        self.kde_is = self.mcmc_kde_computation()
        self.df_is = self.importance_simulation()

        # print(np.mean(self.df["result"]))
        # print(np.mean(self.df_is["result"]*self.df_is["density_orig"]/self.df_is["density_is"]) *
        #       np.mean(self.df["tries"]) / np.mean(self.df_is["tries"]))
        # print(np.mean(self.df["tries"]) / np.mean(self.df_is["tries"]))

    def get_kde(self) -> KDE:
        """ Return the KDE for the braking parameters.

        For now, the scenarios with the ego vehicle braking are used. This should
        later be changed to scenarios with the lead vehicle braking.

        :return: KDE of the braking parameters.
        """
        filename_kde = os.path.join("data", "6_kde", self.options.filename_kde_pars)
        if os.path.exists(filename_kde) and not self.options.overwrite:
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
            print("KDE bandwidth: {:.3f}".format(kde.bandwidth))

            # Write the KDE through a JSON file.
            kde_to_json(filename_kde, kde)

        return kde

    def initial_simulation(self) -> pd.DataFrame:
        """ Do the initial simulation based on the probided grid.

        :return: The results of the simulations.
        """
        filename_df = os.path.join("data", "7_simulation_results", self.options.filename_df)

        if os.path.exists(filename_df) and not self.options.overwrite:
            return pd.read_csv(filename_df, index_col=0)

        # Go through the grid while storing the parameters and the result.
        pars = np.zeros((np.prod([len(grid) for grid in self.options.grid_parameters]),
                         len(self.options.parameters)))
        valid = np.zeros(pars.shape[0], dtype=np.bool)
        result = np.zeros(pars.shape[0])
        print("Simulations with parameters on a grid")
        for i, parameters in enumerate(tqdm(itertools.product(*self.options.grid_parameters),
                                            total=pars.shape[0])):
            par_dict = dict(zip(self.options.parameters, parameters))
            par_dict.update(self.options.default_parameters)
            pars[i] = parameters
            valid[i] = self.options.func_validity_parameters(parameters)
            if valid[i]:
                result[i] = self.options.simulator.get_probability(par_dict, seed=self.options.seed)

        df = pd.DataFrame(data=pars, columns=self.options.parameters)
        df["valid"] = valid
        df["result"] = result

        # Write to file.
        if not os.path.exists(os.path.dirname(filename_df)):
            os.mkdir(os.path.dirname(filename_df))
        df.to_csv(filename_df)

        return df

    def initial_mc_simulation(self) -> pd.DataFrame:
        """ Do the initial Monte Carlo simulations.

        :return: The resulting probabilities of failures (P(B)).
        """
        filename_df = os.path.join("data", "7_simulation_results", self.options.filename_mc)

        if os.path.exists(filename_df) and not self.options.overwrite:
            return pd.read_csv(filename_df, index_col=0)

        # Generate the parameters.
        np.random.seed(0)
        pars = np.zeros((self.options.nmc, len(self.options.parameters)))
        tries = np.zeros(self.options.nmc)
        i = 0
        while i < self.options.nmc:
            pars[i, :] = self.kde.sample()
            tries[i] += 1
            if self.options.func_validity_parameters(pars[i, :]):
                i += 1
        df = pd.DataFrame(data=pars, columns=self.options.parameters)
        df["tries"] = tries

        # Loop through each parameter vector and obtain the simulation result.
        result = np.zeros(self.options.nmc)
        print("Initial Monte Carlo simulations")
        for i, par in enumerate(tqdm(pars)):
            par_dict = dict(zip(self.options.parameters, par))
            par_dict.update(self.options.default_parameters)
            result[i] = self.options.simulator.get_probability(par_dict, seed=self.options.seed)
        df["result"] = result

        # Write to file.
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

    def mcmc_kde_computation(self) -> KDE:
        """ Perform the Markov Chain Monte Carlo to obtain a KDE of the importance density.

        :return: The resulting KDE of the importance density.
        """
        filename_kde = os.path.join("data", "6_kde", self.options.filename_kde_mcmc)
        if os.path.exists(filename_kde) and not self.options.overwrite:
            return kde_from_json(filename_kde)

        np.random.seed(0)
        # Check if initial point is a valid scenario.
        par = np.array(self.options.init_par_mcmc)
        if not self.options.func_validity_parameters(par):
            raise ValueError("Initial parameters for MCMC are not valid.")

        # Create the list of the parameters for the importance density.
        init_pars = self.df[self.options.parameters].values
        pars = np.zeros((self.options.nmcmc, len(self.options.parameters)))
        is_current = self.prob_is(init_pars, par/self.kde.data_helpers.std)
        print("Monte Carlo Markov Chain")
        for i in tqdm(range(-self.options.nburnin,
                            (self.options.nmcmc-1)*self.options.nthinning+1)):
            while True:
                candidate = par + (np.random.randn(len(self.options.parameters))*
                                   self.options.mcmc_step*self.kde.data_helpers.std)
                if self.options.func_validity_parameters(candidate):
                    break
            is_candidate = self.prob_is(init_pars, candidate)
            if is_candidate > 0.0 and np.random.rand()*is_current < is_candidate:
                par = candidate.copy()
                is_current = is_candidate
            if i >= 0 and i % self.options.nthinning == 0:
                pars[i // self.options.nthinning, :] = par

        # Create the KDE.
        kde = KDE(pars, scaling=True)
        kde.compute_bandwidth()
        print("KDE bandwidth: {:.3f}".format(kde.bandwidth))

        # Write the KDE through a JSON file.
        kde_to_json(filename_kde, kde)

        return kde

    def importance_simulation(self) -> pd.DataFrame:
        """ Do the simulations based on the importance sampling.

        For each simulation, the following results are stored:
        1. The number of scenarios that were generated until a valid scenario
           has been generated.
        2. The resulting probability of failure.
        3. The density of the original density.
        4. The density of the importance density.

        :return: The resulting probabilities of failures (P(B)).
        """
        filename_df = os.path.join("data", "7_simulation_results", self.options.filename_dfis)

        if os.path.exists(filename_df) and not self.options.overwrite:
            return pd.read_csv(filename_df, index_col=0)

        # Generate the parameters.
        np.random.seed(0)
        pars = np.zeros((self.options.nsimis, len(self.options.parameters)))
        tries = np.zeros(self.options.nsimis)
        i = 0
        while i < self.options.nsimis:
            pars[i, :] = self.kde_is.sample()
            tries[i] += 1
            if self.options.func_validity_parameters(pars[i, :]):
                i += 1
        df = pd.DataFrame(data=pars, columns=self.options.parameters)
        df["tries"] = tries

        # Obtain the original density and the importance density.
        df["density_orig"] = self.kde.score_samples(pars)
        df["density_is"] = self.kde_is.score_samples(pars)

        # Loop through each parameter vector and obtain the simulation result.
        result = np.zeros(self.options.nsimis)
        print("Simulations with importance density")
        for i, par in enumerate(tqdm(pars)):
            par_dict = dict(zip(self.options.parameters, par))
            par_dict.update(self.options.default_parameters)
            result[i] = self.options.simulator.get_probability(par_dict, seed=self.options.seed)
        df["result"] = result

        # Write to file.
        df.to_csv(filename_df)

        return df
