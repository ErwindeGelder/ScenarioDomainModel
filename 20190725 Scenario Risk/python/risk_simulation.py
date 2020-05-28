import matplotlib.pyplot as plt
import numpy as np
from simulation import IDM, IDMParameters, LeaderBraking, LeaderBrakingParameters

L = LeaderBraking()
F = IDM()


def simulation(v0, amean, dv, plot=False):
    t = 0
    prev_dist = 0
    xl, xf = 1, 0
    deltat = 0.01

    F.init_simulation(IDMParameters(free_speed=10*v0,
                                    init_speed=v0,
                                    dt=deltat,
                                    n_reaction=0))
    L.init_simulation(LeaderBrakingParameters(init_position=(F.parms.safety_distance +
                                                             v0 * F.parms.thw),
                                              init_speed=v0,
                                              average_deceleration=amean,
                                              speed_difference=dv))

    x_and_v = []
    collision = 0

    while t < 10 or prev_dist > xl - xf:
        prev_dist = xl - xf
        t += deltat
        xl, vl = L.step_simulation(t)
        xf, vf = F.step_simulation(xl, vl)

        if plot:
            x_and_v.append([xl, xf, vl, vf])

        if xf > xl:
            collision = 1
            break

    if plot:
        x_and_v = np.array(x_and_v)
        time = np.arange(len(x_and_v)) * deltat
        plt.plot(time, x_and_v[:, 0]-x_and_v[:, 1])
        plt.xlabel("Time [s]")
        plt.ylabel("Distance [m]")
        plt.show()

    return collision


v0 = 30
amean = 2
dv = 20
print(simulation(v0, amean, dv, plot=True))
