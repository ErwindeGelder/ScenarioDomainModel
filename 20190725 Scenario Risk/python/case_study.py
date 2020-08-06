""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
2020 06 08 Perform the case studies in a separate file.
2020 06 16 Do the initial simulations on a grid instead of via Monte Carlo.
2020 08 06 Update the way the case study is performed. Directly MCMC instead of first grid sampling.
"""

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

    filename_kde_pars: str = ""
    filename_mc: str = ""
    filename_is: str = ""
    filename_kde_is: str = ""

    func_parameters: Callable = None
    func_kde_update = None
    func_validity_parameters: Callable = None
    func_process_result: Callable = None

    simulator: Simulator = None

    parameters: List[str] = []
    default_parameters: dict = dict()
    init_par_is: List[float] = []

    mcmc_step: Union[float, np.ndarray] = 0.5

    seed: int = 0
    nmc: int = 1000
    nis: int = 1000
    nburnin: int = 10
    ntestrejection: int = 1000


class CaseStudy:
    """ Object for performing a case study. """
    def __init__(self, options: CaseStudyOptions):
        self.options = options
        self.kde = self.get_kde()
        if self.options.func_kde_update is not None:
            self.kde = self.options.func_kde_update(self.kde, overwrite=self.options.overwrite)
        self.df_mc = self.mc_simulation()
        self.mc_result()
        self.df_is = self.importance_simulation()
        self.kde_is = self.is_result()

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
            kde = kde_from_file(filename_kde)
        else:
            pars = self.options.func_parameters()
            kde = KDE(pars, scaling=True)
            kde.compute_bandwidth()
            print("KDE bandwidth: {:.3f}".format(kde.bandwidth))
            kde.pickle(filename_kde)

        return kde

    def mc_simulation(self) -> pd.DataFrame:
        """ Do the initial Monte Carlo simulations.

        :return: The results from the simulation.
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
        kpi = np.zeros(self.options.nmc)
        print("Initial Monte Carlo simulations")
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

    def mc_result(self) -> None:
        """ Print the Monte Carlo result. """
        prob = np.mean(self.df_mc["kpi"])
        sigma = np.sqrt(np.mean((self.df_mc["kpi"] - prob)**2) / self.options.nmc)
        print("Monte Carlo:         Probability of collision: {:.2f} %".format(prob*100), end="")
        print(" +/- {:.2f} %".format(sigma*100))

    def importance_simulation(self) -> pd.DataFrame:
        """ Do the simulations based on the importance sampling.

        :return: The results from the simulation.
        """
        filename_df = os.path.join("data", "7_simulation_results", self.options.filename_is)

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
        rng = np.random.RandomState(0)
        pars = []
        results = []
        kpis = []
        for i in tqdm(range(-self.options.nburnin, self.options.nis)):
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
                    result_candidate = self.options.simulator.simulation(par_dict,
                                                                         seed=self.options.seed)
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

    def is_result(self) -> KDE:
        """ Print the result of the importance sampling.

        :return: The KDE of the importance density.
        """
        # Compute the importance density.
        filename = os.path.join("data", "6_kde", self.options.filename_kde_is)
        if os.path.exists(filename) and not self.options.overwrite:
            kde_is = kde_from_file(filename)
        else:
            kde_is = KDE(self.df_is[self.options.parameters].values, scaling=True)
            kde_is.clustering()
            kde_is.compute_bandwidth()
            kde_is.pickle(filename)

        # Compute the probability density of the original and the importance density.
        if "orig_density" not in self.df_is.columns or "impo_density" not in self.df_is.columns:
            np.random.seed(0)
            orig_density = self.kde.score_samples(self.df_is[self.options.parameters].values)
            samples = self.kde.sample(self.options.ntestrejection)
            acceptance = np.mean([self.options.func_validity_parameters(sample)
                                  for sample in samples])
            orig_density /= acceptance
            impo_density = kde_is.score_samples(self.df_is[self.options.parameters].values)
            samples = kde_is.sample(self.options.ntestrejection)
            acceptance = np.mean([self.options.func_validity_parameters(sample)
                                  for sample in samples])
            impo_density /= acceptance

            self.df_is["orig_density"] = orig_density
            self.df_is["impo_density"] = impo_density
            self.df_is.to_csv(os.path.join("data", "7_simulation_results",
                                           self.options.filename_is))

        # Compute the Monte Carlo result
        result = self.df_is["kpi"] * self.df_is["orig_density"] / self.df_is["impo_density"]
        prob = np.mean(result)
        sigma = np.sqrt(np.mean((result - prob)**2) / len(result))
        print("Importance sampling: Probability of collision: {:.2f} %".format(prob*100), end="")
        print(" +/- {:.2f} %".format(sigma*100))

        return kde_is
