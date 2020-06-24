""" Testing the scripts of the simulation library.

Creation date: 2020 06 23
Author(s): Erwin de Gelder

Modifications:
"""

import matplotlib.pyplot as plt
from simulation import EIDMPlus, EIDMParameters, SimulationLeadBraking


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
                                dt=0.01,
                                n_reaction=0,
                                thw=1.1,
                                safety_distance=2,
                                amin=-3,
                                a_acc=1,
                                b_acc=1.5,
                                coolness=0.99)
    return parameters


if __name__ == "__main__":
    # Perform simulation with the HDM as follower.
    SIMULATOR = SimulationLeadBraking()
    SIMULATOR.simulation(dict(v0=20, amean=1.8, dv=18), plot=True, seed=0)

    # Same, but estimate probability of a collision. Use two different seeds.
    SIMULATOR.get_probability(dict(v0=20, amean=1.8, dv=18), plot=True, seed=0)
    OLD_TITLE = plt.gca().get_title()
    SIMULATOR.get_probability(dict(v0=20, amean=1.8, dv=18), plot=plt.gca(), seed=3)
    plt.title("1: {:s}\n2: {:s}".format(OLD_TITLE, plt.gca().get_title()))

    # Perform simulation with EIDM+ as follower.
    SIMULATOR = SimulationLeadBraking(follower=EIDMPlus(),
                                      follower_parameters=eidm_parameters,
                                      stochastic=False)
    SIMULATOR.simulation(dict(v0=20, amean=1.8, dv=18), plot=True, seed=0)

    plt.show()
