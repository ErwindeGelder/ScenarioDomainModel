""" Simulation of the scenario "lead vehicle braking".

Creation date: 2020 05 29
Author(s): Erwin de Gelder

Modifications:
"""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from .hdm import HDM, HDMParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_braking import LeaderBraking, LeaderBrakingParameters
from .simulator import Simulator


class SimulationLeadBraking(Simulator):
    """ Class for simulation the scenario "lead vehicle braking".

    Attributes:
        leader(LeaderBraking)
        follower(HDM)
        follower_parameters(HDMParameters)
    """
    def __init__(self, **kwargs):
        # Instantiate the vehicles.
        self.leader = LeaderBraking()
        self.follower = HDM()
        self.follower_parameters = HDMParameters(model=IDMPlus(), speed_std=0.05, tau=20, rttc=0.01,
                                                 dt=0.01)
        Simulator.__init__(self, **kwargs)

    def simulation(self, parameters: Tuple[float, float, float], plot: bool = False,
                   seed: int = None) -> float:
        """ Run a single simulation.

        :param parameters: (starting speed, average deceleration, speed difference)
        :param plot: Whether to make a plot or not.
        :param seed: Specify in order to simulate with a fixed seed.
        :return: The minimum distance (negative means collision).
        """
        if seed is not None:
            np.random.seed(seed)
        self.init_simulation(parameters)
        time = 0
        prev_dist = 0
        x_leader, x_follower = 100, 0

        data = []
        mindist = 100

        # Run the simulation for at least 10 seconds. Stop the simulation if the
        # distance increases.
        while time < 10 or prev_dist > x_leader - x_follower:
            prev_dist = x_leader - x_follower
            mindist = min(prev_dist, mindist)
            time += self.follower.parms.dt
            x_leader, v_leader = self.leader.step_simulation(time)
            x_follower, v_follower = self.follower.step_simulation(x_leader, v_leader)

            if plot:
                data.append([x_leader, x_follower, v_leader, v_follower,
                             self.leader.state.acceleration,
                             self.follower.parms.model.state.acceleration])

            if time > 100:
                break

        if plot:
            _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
            data = np.array(data)
            time = np.arange(len(data)) * self.follower.parms.dt
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
            plt.show()

        return mindist

    def init_simulation(self, parameters: Tuple[float, float, float]) -> None:
        """ Initialize the simulation.

        :param parameters: (starting speed, average deceleration, speed difference)
        """
        init_speed, average_deceleration, speed_difference = parameters
        self.follower.init_simulation(self.follower_parameters,
                                      IDMParameters(free_speed=init_speed * 1.2,
                                                    init_speed=init_speed,
                                                    dt=self.follower_parameters.dt,
                                                    n_reaction=100,
                                                    thw=1,
                                                    safety_distance=2,
                                                    amin=-3))
        self.leader.init_simulation(
            LeaderBrakingParameters(init_position=(self.follower.parms.model.parms.safety_distance +
                                                   init_speed*self.follower.parms.model.parms.thw),
                                    init_speed=init_speed,
                                    average_deceleration=average_deceleration,
                                    speed_difference=speed_difference,
                                    tconst=5))
