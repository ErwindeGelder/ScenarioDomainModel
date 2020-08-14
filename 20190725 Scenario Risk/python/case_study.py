""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
2020 06 08 Perform the case studies in a separate file.
2020 06 16 Do the initial simulations on a grid instead of via Monte Carlo.
2020 08 06 Update the way the case study is performed. Directly MCMC instead of first grid sampling.
2020 08 11 Also do importance sampling with a meta model based on grid simulations.
"""

import itertools
import os
from typing import Callable, List, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from simulation import Simulator
from stats import KDE, kde_from_file
from options import Options


def default_process_result(result):
    """ Returns a True if result is negative.

    :param result: The result from a simulation.
    :return: Key performance indicator.
    """
    return result < 0


class CaseStudyOptions(Options):
    """ Options for a case study. """
    overwrite: bool = False

    filename_prefix: str = ""

    func_parameters: Callable = None
    func_kde_update = None
    func_validity_parameters: Callable = None
    func_process_result: Callable = None

    simulator: Simulator = None

    parameters: List[str] = []
    grid_parameters: List[np.ndarray] = []
    default_parameters: dict = dict()
    init_par_is: List[float] = []

    mcmc_step: Union[float, np.ndarray] = 0.5

    seed: int = 0
    nmc: int = 1000
    nburnin: int = 100
    nmcmc: int = 1000
    nthinning: int = 100
    nis: int = 1000
    nis_direct: int = 1000
    ntestrejection: int = 1000


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
        self.df_mc = self.mc_simulation()
        self.mc_result()
        self.df_grid = self.grid_simulation()
        self.kde_is = self.mcmc_kde_computation()
        self.df_is = self.importance_simulation()
        self.is_result()
        self.df_is_direct = self.direct_importance_simulation()
        self.kde_is_direct = self.direct_is_result()

    def get_kde(self) -> KDE:
        """ Return the KDE for the braking parameters.

        For now, the scenarios with the ego vehicle braking are used. This should
        later be changed to scenarios with the lead vehicle braking.

        :return: KDE of the braking parameters.
        """
        filename_kde = os.path.join("data", "6_kde", "{:s}.p".format(self.options.filename_prefix))
        if os.path.exists(filename_kde) and not self.options.overwrite:
            kde = kde_from_file(filename_kde)
        else:
            pars = self.options.func_parameters()
            kde = KDE(pars, scaling=True)
            kde.compute_bandwidth()
            print("KDE bandwidth: {:.3f}".format(kde.bandwidth))

            if self.options.func_kde_update is not None:
                kde = self.options.func_kde_update(kde)

            kde.pickle(filename_kde)

        return kde

    def mc_simulation(self) -> pd.DataFrame:
        """ Do the initial Monte Carlo simulations.

        :return: The results from the simulation.
        """
        filename_df = os.path.join("data", "7_simulation_results",
                                   "{:s}_mc.csv".format(self.options.filename_prefix))

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
        kpi = np.zeros(self.options.nmc)
        print("Initial Monte Carlo simulations")
        for i, par in enumerate(tqdm(pars)):
            par_dict = dict(zip(self.options.parameters, par))
            par_dict.update(self.options.default_parameters)
            result[i] = self.options.simulator.simulation(par_dict)
            kpi[i] = self.options.func_process_result(result[i])
        df["result"] = result
        df["kpi"] = kpi

        # Write to file.
        df.to_csv(filename_df)

        return df

    def mc_result(self) -> None:
        """ Print the Monte Carlo result. """
        prob = np.mean(self.df_mc["kpi"])
        sigma = np.sqrt(np.sum((self.df_mc["kpi"] - prob)**2)) / len(self.df_mc)
        print("Monte Carlo:         Probability of collision: {:.3e} %".format(prob), end="")
        print(" +/- {:.3e} %".format(sigma), end="")
        print(" ({:d} simulations)".format(len(self.df_mc)))

    def grid_simulation(self) -> pd.DataFrame:
        """ Do the simulations based on a grid.

        :return: The results from the simulation.
        """
        filename_df = os.path.join("data", "7_simulation_results",
                                   "{:s}_grid.csv".format(self.options.filename_prefix))

        if os.path.exists(filename_df) and not self.options.overwrite:
            return pd.read_csv(filename_df, index_col=0)

        # Go through the grid while storing the parameters and the result.
        np.random.seed(0)
        pars = np.zeros((np.prod([len(grid) for grid in self.options.grid_parameters]),
                         len(self.options.parameters)))
        valid = np.zeros(pars.shape[0], dtype=np.bool)
        result = np.zeros(pars.shape[0])
        kpi = np.zeros(pars.shape[0])
        print("Simulations with parameters on a grid")
        for i, parameters in enumerate(tqdm(itertools.product(*self.options.grid_parameters),
                                            total=pars.shape[0])):
            par_dict = dict(zip(self.options.parameters, parameters))
            par_dict.update(self.options.default_parameters)
            pars[i] = parameters
            valid[i] = self.options.func_validity_parameters(parameters)
            if valid[i]:
                result[i] = self.options.simulator.simulation(par_dict)
                kpi[i] = self.options.func_process_result(result[i])

        df = pd.DataFrame(data=pars, columns=self.options.parameters)
        df["valid"] = valid
        df["result"] = result
        df["kpi"] = kpi

        # Write to file.
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
        kpi = self.df_grid.at[i, "kpi"]
        if kpi == 0.0:
            return 0
        return kpi*self.kde.score_samples(np.array([par]))[0]

    def mcmc_kde_computation(self) -> KDE:
        """ Perform the Markov Chain Monte Carlo to obtain a KDE of the importance density.

        :return: The resulting KDE of the importance density.
        """
        filename_kde = os.path.join("data", "6_kde",
                                    "{:s}_is.p".format(self.options.filename_prefix))
        if os.path.exists(filename_kde) and not self.options.overwrite:
            return kde_from_file(filename_kde)

        np.random.seed(0)
        # Check if initial point is a valid scenario.
        par = np.array(self.options.init_par_is)
        if not self.options.func_validity_parameters(par):
            raise ValueError("Initial parameters for MCMC are not valid.")

        # Create the list of the parameters for the importance density.
        init_pars = self.df_grid[self.options.parameters].values
        pars = np.zeros((self.options.nmcmc, len(self.options.parameters)))
        is_current = self.prob_is(init_pars, par/self.kde.data_helpers.std)
        print("Monte Carlo Markov Chain")
        for i in tqdm(range(-self.options.nburnin,
                            (self.options.nmcmc-1)*self.options.nthinning+1)):
            while True:
                candidate = par + (np.random.randn(len(self.options.parameters)) *
                                   self.options.mcmc_step*self.kde.data_helpers.std)
                if self.options.func_validity_parameters(candidate):
                    break
            is_candidate = self.prob_is(init_pars, candidate)
            if is_candidate > 0.0 and np.random.rand()*is_current < is_candidate:
                par = candidate.copy()
                is_current = is_candidate
            if i >= 0 and i % self.options.nthinning == 0:
                pars[i//self.options.nthinning, :] = par

        # Create the KDE.
        kde = KDE(pars, scaling=True)
        kde.clustering()
        kde.compute_bandwidth()
        print("KDE bandwidth: {:.3f}".format(kde.bandwidth))
        kde.pickle(filename_kde)
        return kde

    def importance_simulation(self) -> pd.DataFrame:
        """ Do the simulations based on the importance sampling.

        For each simulation, the following results are stored:
        1. The number of scenarios that were generated until a valid scenario
           has been generated.
        2. The result of the simulation.
        3. The corresponding KPI.
        4. The density of the original density.
        5. The density of the importance density.

        :return: The resulting dataframe
        """
        filename_df = os.path.join("data", "7_simulation_results",
                                   "{:s}_is.csv".format(self.options.filename_prefix))

        if os.path.exists(filename_df) and not self.options.overwrite:
            return pd.read_csv(filename_df, index_col=0)

        # Generate the parameters.
        np.random.seed(0)
        pars = np.zeros((self.options.nis, len(self.options.parameters)))
        tries = np.zeros(self.options.nis)
        i = 0
        while i < self.options.nis:
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
        result = np.zeros(self.options.nis)
        kpi = np.zeros(self.options.nis)
        print("Simulations with importance density")
        for i, par in enumerate(tqdm(pars)):
            par_dict = dict(zip(self.options.parameters, par))
            par_dict.update(self.options.default_parameters)
            result[i] = self.options.simulator.simulation(par_dict, seed=self.options.seed)
            kpi[i] = self.options.func_process_result(result[i])
        df["result"] = result
        df["kpi"] = kpi

        # Write to file.
        df.to_csv(filename_df)

        return df

    def is_result(self) -> None:
        """ Show the result from the importance sampling. """
        values = self.df_is["kpi"] * self.df_is["density_orig"] / self.df_is["density_is"]
        values *= np.sum(self.df_mc["tries"]) / len(self.df_mc)
        values /= np.sum(self.df_is["tries"]) / len(self.df_is)
        prob = np.mean(values)
        sigma = np.sqrt(np.sum((values - prob)**2)) / len(values)
        print("Importance sampling: Probability of collision: {:.3e} %".format(prob), end="")
        print(" +/- {:.3e} %".format(sigma), end="")
        print(" ({:d} simulations)".format(len(values)))

    def direct_importance_simulation(self) -> pd.DataFrame:
        """ Do the simulations based on the importance sampling.

        :return: The results from the simulation.
        """
        filename_df = os.path.join("data", "7_simulation_results",
                                   "{:s}_is_direct.csv".format(self.options.filename_prefix))

        if os.path.exists(filename_df) and not self.options.overwrite:
            return pd.read_csv(filename_df, index_col=0)

        # Check if the starting parameters are valid.
        pars_current = self.options.init_par_is
        par_dict = dict(zip(self.options.parameters, pars_current))
        par_dict.update(self.options.default_parameters)
        result = self.options.simulator.simulation(par_dict, seed=self.options.seed)
        kpi = self.options.func_process_result(result)
        if kpi <= 0:
            raise ValueError("Simulation should give strictly positive result for initial values.")
        current_prob = self.kde.score_samples(np.array([pars_current]))

        # Do the importance sampling using MCMC.
        np.random.seed(0)
        rng = np.random.RandomState(0)
        pars = []
        results = []
        kpis = []
        for i in tqdm(range(-self.options.nburnin, self.options.nis_direct)):
            simulation_done = False
            while True:
                pars_candidate = (pars_current + rng.randn(len(self.options.parameters)) *
                                  self.options.mcmc_step)
                if not self.options.func_validity_parameters(pars_candidate):
                    candidate_prob = 0
                else:
                    candidate_prob = self.kde.score_samples(np.array([pars_candidate]))
                prob_mcmc = np.random.rand()
                if prob_mcmc < candidate_prob / current_prob / kpi:
                    simulation_done = True
                    par_dict = dict(zip(self.options.parameters, pars_candidate))
                    par_dict.update(self.options.default_parameters)
                    result_candidate = self.options.simulator.simulation(par_dict)
                    kpi_candidate = self.options.func_process_result(result_candidate)
                    if prob_mcmc < candidate_prob / current_prob / kpi * kpi_candidate:
                        pars_current = pars_candidate.copy()
                        current_prob = candidate_prob
                        result = result_candidate
                        kpi = kpi_candidate

                if i >= 0:
                    pars.append(pars_current)
                    results.append(result)
                    kpis.append(kpi)

                if simulation_done:
                    break

        df = pd.DataFrame(data=pars, columns=self.options.parameters)
        df["result"] = results
        df["kpi"] = kpis
        df.to_csv(filename_df)
        return df

    def direct_is_result(self) -> KDE:
        """ Print the result of the importance sampling.

        :return: The KDE of the importance density.
        """
        # Compute the importance density.
        filename = os.path.join("data", "6_kde",
                                "{:s}_is_direct.p".format(self.options.filename_prefix))
        if os.path.exists(filename) and not self.options.overwrite:
            kde_is = kde_from_file(filename)
        else:
            kde_is = KDE(self.df_is_direct[self.options.parameters].values, scaling=True)
            kde_is.clustering()
            kde_is.compute_bandwidth()
            kde_is.pickle(filename)

        # Compute the probability density of the original and the importance density.
        if "orig_density" not in self.df_is_direct.columns or \
                "impo_density" not in self.df_is_direct.columns:
            np.random.seed(0)
            orig_density = self.kde.score_samples(self.df_is_direct[self.options.parameters].values)
            samples = self.kde.sample(self.options.ntestrejection)
            acceptance = np.mean([self.options.func_validity_parameters(sample)
                                  for sample in samples])
            orig_density /= acceptance
            impo_density = kde_is.score_samples(self.df_is_direct[self.options.parameters].values)
            samples = kde_is.sample(self.options.ntestrejection)
            acceptance = np.mean([self.options.func_validity_parameters(sample)
                                  for sample in samples])
            impo_density /= acceptance

            self.df_is_direct["orig_density"] = orig_density
            self.df_is_direct["impo_density"] = impo_density
            self.df_is_direct.to_csv(os.path.join("data", "7_simulation_results",
                                                  "{:s}_is_direct.csv".
                                                  format(self.options.filename_prefix)))

        # Compute the Monte Carlo result
        result = (self.df_is_direct["kpi"] * self.df_is_direct["orig_density"] /
                  self.df_is_direct["impo_density"])
        prob = np.mean(result)
        sigma = np.sqrt(np.sum((result - prob)**2)) / len(result)
        print("Alternative IS:      Probability of collision: {:.3e} %".format(prob), end="")
        print(" +/- {:.3e} %".format(sigma), end="")
        print(" ({:d} simulations)".format(len(result)))

        return kde_is
