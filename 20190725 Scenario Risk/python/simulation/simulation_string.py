""" Class for simulations of longitudinal scenarios with n vehicles.

Creation date: 2020 12 09
Author(s): Erwin de Gelder

Modifications:
2020 12 11: Return negative impact speed in case of a collision.
"""

from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
from .simulator import Simulator


class SimulationString(Simulator):
    """ Class for simulation of longitudinal scenarios with a leading vehicle

    Attributes:
        vehicles - list of vehicles
        parameters - list of functions for obtaining the parameters
    """
    def __init__(self, vehicles: List, parameter_functions: List[Callable], **kwargs):
        # Instantiate the vehicles.
        self.vehicles = vehicles
        self.parameter_functions = parameter_functions
        Simulator.__init__(self, **kwargs)

    def simulation(self, parameters: dict, plot: bool = False, ignore_stop: List[bool] = None,
                   seed: int = None) -> np.ndarray:
        """ Run a single simulation.

        :param parameters: specific parameters for the scenario.
        :param plot: Whether to make a plot or not.
        :param ignore_stop: Whether to ignore stopping criterium for a vehicle.
        :param seed: Specify in order to simulate with a fixed seed.
        :return: The minimum distance (negative means collision).
        """
        if seed is not None:
            np.random.seed(seed)
        if ignore_stop is None:
            ignore_stop = np.zeros(len(self.vehicles)-1, dtype=np.bool)
        self.init_simulation(**parameters)
        time = 0

        data = []

        # Run the simulation for at least some time. Stop the simulation if the distance has been
        # increasing between all vehicles or all vehicles impacted.
        distance_increasing = ignore_stop.copy()
        distances = np.ones(len(self.vehicles)-1)*1000
        mindistances = distances.copy()
        minttcs = distances.copy()
        impact_speeds = np.zeros(len(self.vehicles)-1)
        while time < self.min_simulation_time or (not np.all(distance_increasing) and
                                                  not np.all(impact_speeds)):
            # Update the distances.
            for i, (leader, follower) in enumerate(zip(self.vehicles[:-1], self.vehicles[1:])):
                distance = leader.state.position - follower.state.position
                if distance < 0 <= distances[i]:  # We have an impact!
                    impact_speeds[i] = follower.state.speed - leader.state.speed
                if distance < mindistances[i]:
                    mindistances[i] = distance
                if not ignore_stop[i]:
                    distance_increasing[i] = distances[i] <= distance + 0.001  # Add some margin
                distances[i] = distance
                if np.abs(leader.state.speed - follower.state.speed) > 1e-5:
                    ttc = ((follower.state.position - leader.state.position) /
                           (leader.state.speed - follower.state.speed))
                    if 0 < ttc < minttcs[i]:
                        minttcs[i] = ttc

            time += self.vehicles[1].parms.timestep
            self.vehicles[0].step_simulation(time)
            for leader, follower in zip(self.vehicles[:-1], self.vehicles[1:]):
                follower.step_simulation(leader)

            if plot:
                data.append([[vehicle.state.position, vehicle.state.speed,
                              vehicle.state.acceleration] for vehicle in self.vehicles])

            if time > 100:
                break

        if plot:
            _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
            data = np.array(data)
            time = np.arange(len(data)) * self.vehicles[1].parms.timestep
            for i in range(len(self.vehicles)-1):
                ax1.plot(time, data[:, i, 0] - data[:, i+1, 0], label="{:d}-{:d}".format(i+1, i+2))
                ax3.plot(time, (data[:, i, 0] - data[:, i+1, 0]) / data[:, i+1, 1],
                         label="{:d}-{:d}".format(i+1, i+2))
            for i in range(len(self.vehicles)):
                ax2.plot(time, data[:, i, 1]*3.6, label="Vehicle {:d}".format(i+1))
                ax4.plot(time, data[:, i, 2], label="Vehicle {:d}".format(i+1))
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Distance [m]")
            ax1.legend()
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Speed [km/h]")
            ax2.legend()
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("THW [s]")
            ax3.set_ylim(-.5, 2)
            ax3.legend()
            ax4.set_xlabel("Time [s]")
            ax4.set_ylabel("Acceleration [m/s$^2$]")
            ax4.legend()
            plt.tight_layout()

        return np.array([minttcs[i] if mindistances[i] > 0 else -impact_speeds[i]
                         for i in range(len(self.vehicles)-1)])

    def init_simulation(self, **kwargs) -> None:
        """ Initialize the simulation.

        :param kwargs: The parameters for the scenario.
        """
        for i, (vehicle, parameter_function) in enumerate(zip(self.vehicles,
                                                              self.parameter_functions)):
            arguments = dict(i=i)
            arguments.update(**kwargs)
            vehicle.init_simulation(parameter_function(**arguments))
