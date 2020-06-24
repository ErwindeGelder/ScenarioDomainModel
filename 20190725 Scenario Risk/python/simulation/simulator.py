""" Provide a class with generic functions for simulations.

Creation date: 2020 05 29
Author(s): Erwin de Gelder

Modifications:
2020 06 18 Make it possible to pass an Axes for plotting with get_probability.
2020 06 24 Add possibility of having a deterministic output (prob=1 or prob=0).
"""

from abc import ABC
from typing import Union
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from .fastkde import KDE


class Simulator(ABC):
    """ Generic functions for a simulator. """
    def __init__(self, tolerance=0.01, min_simulations=5, max_simulations=100, stochastic=True):
        self.tolerance = tolerance
        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.stochastic = stochastic

    def simulation(self, parameters: dict, plot=False, seed: int = None):
        """ Run a single simulation.

        :param parameters: Parameters of the simulation.
        :param plot: Whether to maka a plot.
        :param seed: Specify in order to simulate with a fixed seed.
        :return: Result of the simulation.
        """

    def get_probability(self, parameters: dict, plot: Union[bool, Axes] = False,
                        seed: int = None) -> float:
        """ Run multiple simulations in order to get the probability of failure.

        :param parameters: Parameters of the simulation.
        :param plot: Whether to make a plot. If so, axes can be provided as well.
        :param seed: Specify in order to simulate with a fixed seed.
        :return: Result of the simulation.
        """
        if not self.stochastic:
            if self.simulation(parameters) > 0:
                return 1
            return 0

        if seed is not None:
            np.random.seed(seed)

        # At least run the simulation few times in order to construct a KDE.
        n_simulations = self.min_simulations
        data = np.array([self.simulation(parameters) for _ in range(n_simulations)])
        kde = KDE(data)
        kde.compute_bandwidth()

        # Keep running simulations until we reached a certainty below the tolerance.
        cdf_zero = kde.cdf(np.array([0.]))[0]
        while np.sqrt(cdf_zero*(1-cdf_zero)/n_simulations) > self.tolerance and \
                n_simulations < self.max_simulations:
            kde.add_data(np.array([self.simulation(parameters)]))
            kde.compute_bandwidth()
            cdf_zero = kde.cdf(np.array([0.]))[0]
            n_simulations += 1

        if plot:
            if isinstance(plot, Axes):
                axes = plot
            else:
                _, axes = plt.subplots(1, 1)
            minx = min(np.min(kde.data), 0) - 2*kde.bandwidth
            maxx = max(np.max(kde.data), 0) + 2*kde.bandwidth
            x_cdf = np.linspace(minx, maxx)
            y_cdf = kde.cdf(x_cdf)
            axes.plot(x_cdf, y_cdf)
            axes.set_xlim(minx, maxx)
            axes.plot(kde.data, np.zeros_like(kde.data), '|')
            axes.set_title("N={:d}, F(0)={:.3f} +/- {:.3f}".format(n_simulations, cdf_zero,
                                                                   np.sqrt(cdf_zero*(1-cdf_zero) /
                                                                           n_simulations)))

        return cdf_zero
