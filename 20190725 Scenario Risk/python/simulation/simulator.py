""" Provide a class with generic functions for simulations.

Creation date: 2020 05 29
Author(s): Erwin de Gelder

Modifications:
"""

from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
from .fastkde import KDE


class Simulator(ABC):
    """ Generic functions for a simulator. """
    def __init__(self, tolerance=0.01, max_simulations=100):
        self.kde = KDE(data=np.array([]))
        self.tolerance = tolerance
        self.max_simulations = max_simulations

    def simulation(self, parameters, plot=False):
        """ Run a single simulation.

        :param parameters: Parameters of the simulation.
        :param plot: Whether to maka a plot.
        :return: Result of the simulation.
        """

    def get_probability(self, parameters, plot=False) -> float:
        """ Run multiple simulations in order to get the probability of failure.

        :param parameters: Parameters of the simulation.
        :param plot: Whether to maka a plot.
        :return: Result of the simulation.
        """
        # At least run the simulation few times in order to construct a KDE.
        n_simulations = 2
        data = np.array([self.simulation(parameters) for _ in range(n_simulations)])
        self.kde.fit(data)
        self.kde.compute_bandwidth()

        # Keep running simulations until we reached a certainty below the tolerance.
        cdf_zero = self.kde.cdf(np.array([0.]))[0]
        while np.sqrt(cdf_zero*(1-cdf_zero)/n_simulations) > self.tolerance and \
                n_simulations < self.max_simulations:
            self.kde.add_data(np.array([self.simulation(parameters)]))
            self.kde.compute_bandwidth()
            cdf_zero = self.kde.cdf(np.array([0.]))[0]
            n_simulations += 1

        if plot:
            minx = min(np.min(self.kde.data), 0) - 2*self.kde.bandwidth
            maxx = max(np.max(self.kde.data), 0) + 2*self.kde.bandwidth
            x_cdf = np.linspace(minx, maxx)
            y_cdf = self.kde.cdf(x_cdf)
            plt.plot(x_cdf, y_cdf)
            plt.xlim(minx, maxx)
            plt.plot(self.kde.data, np.zeros_like(self.kde.data), '|')
            plt.title("N={:d}, F(0)={:.3f} +/- {:.3f}".format(n_simulations, cdf_zero,
                                                              np.sqrt(cdf_zero*(1-cdf_zero) /
                                                                      n_simulations)))
            plt.show()

        return cdf_zero
