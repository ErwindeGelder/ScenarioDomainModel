""" Simulation of the scenario "lead vehicle braking".

Creation date: 2020 05 29
Author(s): Erwin de Gelder

Modifications:
2020 06 22 Parameters based on Treiber et al. (2006).
2020 06 23 Add possibility to use any driver model for follower.
"""

import matplotlib.pyplot as plt
import numpy as np
from .acc import ACC, ACCParameters
from .eidm import EIDMParameters
from .hdm import HDM, HDMParameters
from .idm import IDMParameters
from .idmplus import IDMPlus
from .leader_braking import LeaderBraking, LeaderBrakingParameters
from .simulator import Simulator


def hdm_parameters(**kwargs):
    """ Define the follower parameters based on the scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    steptime = 0.01
    safety_distance = 2.0
    thw = 1.1
    init_distance = safety_distance + init_speed * thw
    parameters = HDMParameters(model=IDMPlus(), speed_std=0.05, tau=20, rttc=0.01,
                               timestep=steptime,
                               parms_model=IDMParameters(speed=init_speed*1.2,
                                                         init_speed=init_speed,
                                                         init_position=-init_distance,
                                                         timestep=0.01,
                                                         n_reaction=100,
                                                         thw=1.1,
                                                         safety_distance=2,
                                                         amin=-8,
                                                         a_acc=1,
                                                         b_acc=1.5))
    return parameters


def eidm_parameters(**kwargs):
    """ Define the follower parameters based on the scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    safety_distance = 2.0
    thw = 1.1
    init_distance = safety_distance + init_speed * thw
    parameters = EIDMParameters(free_speed=init_speed*1.2,
                                init_speed=init_speed,
                                init_position=-init_distance,
                                timestep=0.01,
                                n_reaction=0,
                                thw=1.1,
                                safety_distance=2,
                                amin=-8,
                                a_acc=1,
                                b_acc=1.5,
                                coolness=0.99)
    return parameters


def acc_parameters(**kwargs):
    """ Define the follower parameters based on the scenario parameters.

    :return: Parameter object that can be passed via init_simulation.
    """
    init_speed = kwargs["v0"]
    safety_distance = ACC.safety_distance(init_speed)
    default_parameters = ACCParameters()
    thw = default_parameters.thw
    init_distance = safety_distance + init_speed * thw
    parameters = ACCParameters(speed=init_speed,
                               init_speed=init_speed,
                               init_position=-init_distance,
                               n_reaction=0)
    return parameters


class SimulationLeadBraking(Simulator):
    """ Class for simulation the scenario "lead vehicle braking".

    Attributes:
        leader(LeaderBraking)
        follower - any given driver model (by default, HDM is used)
        follower_parameters - function for obtaining the parameters.
    """
    def __init__(self, follower=None, follower_parameters=None, **kwargs):
        # Instantiate the vehicles.
        self.leader = LeaderBraking()
        self.follower = HDM() if follower is None else follower
        if follower_parameters is None:
            self.follower_parameters = hdm_parameters
        else:
            self.follower_parameters = follower_parameters
        Simulator.__init__(self, **kwargs)

    def simulation(self, parameters: dict, plot: bool = False,
                   seed: int = None) -> float:
        """ Run a single simulation.

        :param parameters: (starting speed, average deceleration, speed difference)
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
        while time < 10 or prev_dist > self.leader.state.position - self.follower.state.position:
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

        :param kwargs: The parameters: v0, amean, dv.
        """
        init_speed = kwargs["v0"]
        average_deceleration = kwargs["amean"]
        speed_difference = kwargs["dv"]
        self.follower.init_simulation(self.follower_parameters(**kwargs))
        self.leader.init_simulation(
            LeaderBrakingParameters(init_position=0,
                                    init_speed=init_speed,
                                    average_deceleration=average_deceleration,
                                    speed_difference=speed_difference,
                                    tconst=5))
