import time
import matplotlib.pyplot as plt
import numpy as np
from simulation import HDM, HDMParameters, IDMParameters, IDMPlus, LeaderBraking,\
    LeaderBrakingParameters

L = LeaderBraking()
F = HDM()
F_PARMS = HDMParameters(model=IDMPlus(), speed_std=0.05, tau=20, rttc=0.01, dt=0.01)
np.random.seed(0)


def simulation(v0, amean, dv, plot=False):
    t = 0
    prev_dist = 0
    xl, xf = 100, 0
    deltat = F_PARMS.dt

    tstart = time.time()
    F.init_simulation(F_PARMS, IDMParameters(free_speed=v0*1.2,
                                             init_speed=v0,
                                             dt=deltat,
                                             n_reaction=100,
                                             thw=1,
                                             safety_distance=0,
                                             amin=-3))
    L.init_simulation(LeaderBrakingParameters(init_position=(F.parms.model.parms.safety_distance +
                                                             v0 * F.parms.model.parms.thw),
                                              init_speed=v0,
                                              average_deceleration=amean,
                                              speed_difference=dv,
                                              tconst=10))

    data = []
    collision = 0
    mindist = 100

    while t < 15 or prev_dist > xl - xf:
        prev_dist = xl - xf
        mindist = min(prev_dist, mindist)
        t += deltat
        xl, vl = L.step_simulation(t)
        xf, vf = F.step_simulation(xl, vl)

        if plot:
            data.append([xl, xf, vl, vf, L.state.acceleration, F.parms.model.state.acceleration])

        if xf > xl:
            collision = 1
            # break
    print(time.time() - tstart)

    print(mindist)
    if plot:
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
        data = np.array(data)
        TIME = np.arange(len(data)) * deltat
        ax1.plot(TIME, data[:, 0]-data[:, 1])
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Distance [m]")
        ax2.plot(TIME, data[:, 2]*3.6, label="lead")
        ax2.plot(TIME, data[:, 3]*3.6, label="host")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Speed [km/h]")
        ax2.legend()
        ax3.plot(TIME, (data[:, 0] - data[:, 1]) / data[:, 3])
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("THW [s]")
        ax4.plot(TIME, data[:, 4], label="lead")
        ax4.plot(TIME, data[:, 5], label="host")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Acceleration [m/s$^2$]")
        ax4.legend()
        plt.tight_layout()
        plt.show()

    return collision


v0 = 30
amean = 3
dv = 20
print(simulation(v0, amean, dv, plot=True))
