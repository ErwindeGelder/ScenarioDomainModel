""" Class for simulations of longitudinal scenarios with a leading vehicle.

Creation date: 2020 08 12
Author(s): Erwin de Gelder

Modifications:
2020 08 14 Minimum simulation time is now an attribute of the object, rather than hardcoded.
"""

import matplotlib.pyplot as plt
import numpy as np
from .simulator import Simulator


class SimulationLongitudinal(Simulator):
    """ Class for simulation of longitudinal scenarios with a leading vehicle

    Attributes:
        leader
        leader_parameters - function for obtaining parameters of leading vehicle
        follower - any given driver model (by default, HDM is used)
        follower_parameters - function for obtaining the parameters
    """
    def __init__(self, leader, leader_parameters, follower, follower_parameters, **kwargs):
        # Instantiate the vehicles.
        self.leader, self.leader_parameters = leader, leader_parameters
        self.follower, self.follower_parameters = follower, follower_parameters
        self.min_simulation_time = 10
        Simulator.__init__(self, **kwargs)

    def simulation(self, parameters: dict, plot: bool = False,
                   seed: int = None) -> float:
        """ Run a single simulation.

        :param parameters: specific parameters for the scenario.
        :param plot: Whether to make a plot or not.
        :param seed: Specify in order to simulate with a fixed seed.
        :return: The minimum distance (negative means collision).
        """
        if seed is not None:
            np.random.seed(seed)
        self.init_simulation(**parameters)
        time = 0
        prev_dist = 0

        data = []
        mindist = 100

        # Run the simulation for at least 10 seconds. Stop the simulation if the
        # distance increases.
        while time < self.min_simulation_time \
                or prev_dist > self.leader.state.position - self.follower.state.position:
            prev_dist = self.leader.state.position - self.follower.state.position
            mindist = min(prev_dist, mindist)
            time += self.follower.parms.timestep
            self.leader.step_simulation(time)
            self.follower.step_simulation(self.leader)

            if plot:
                data.append([self.leader.state.position, self.follower.state.position,
                             self.leader.state.speed, self.follower.state.speed,
                             self.leader.state.acceleration,
                             self.follower.state.acceleration])

            if time > 100:
                break

        if plot:
            _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
            data = np.array(data)
            time = np.arange(len(data)) * self.follower.parms.timestep
            ax1.plot(time, data[:, 0] - data[:, 1])
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Distance [m]")
            ax2.plot(time, data[:, 2] * 3.6, label="lead")
            ax2.plot(time, data[:, 3] * 3.6, label="host")
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Speed [km/h]")
            ax2.legend()
            ax3.plot(time, (data[:, 0] - data[:, 1]) / data[:, 3])
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("THW [s]")
            ax4.plot(time, data[:, 4], label="lead")
            ax4.plot(time, data[:, 5], label="host")
            ax4.set_xlabel("Time [s]")
            ax4.set_ylabel("Acceleration [m/s$^2$]")
            ax4.legend()
            plt.tight_layout()

        return mindist

    def init_simulation(self, **kwargs) -> None:
        """ Initialize the simulation.

        :param kwargs: The parameters for the scenario.
        """
        self.follower.init_simulation(self.follower_parameters(**kwargs))
        self.leader.init_simulation(self.leader_parameters(**kwargs))
