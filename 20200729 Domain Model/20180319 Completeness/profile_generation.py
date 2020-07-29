import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


class ProfileGeneration:
    def __init__(self):
        # Load the profiles
        with open(os.path.join('pickles', 'df.p'), 'rb') as f:
            self.dfs, self.scaling = pickle.load(f)

        self.ts = 1 / 10  # [s]  Frequency
        self.transition = np.array([[0, 0.8, 0.2, 0],  # braking
                                    [0.5, 0, 0.5, 0],  # cruising
                                    [0.3, 0.7, 0, 0],  # accelerating
                                    [0, 0, 1, 0]])  # full stop

        # Braking parameters
        self.brk_shape = 1  # [m/s]
        self.brk_std = 2  # [m/s]
        self.min_vel = 5  # [m/s]  If velocity is below this speed, then it is set to 0
        self.min_vel_reduction = 3  # [m/s]
        self.decelerations = self.scaling[0] / self.scaling[1]
        self.p_full_stop = 0.1

        # Cruising parameters
        self.min_cruising_time = 3  # [s]
        self.max_cruising_time = 17  # [s]
        self.max_vel_difference = 0  # 0.3  # [m/s]

        # Acceleration parameters
        self.acc_shape = 1  # [m/s] Shape velocity increase
        self.acc_std = 2  # [m/s] Standard deviation velocity increase
        self.max_vel_accelerating = 130 / 3.6  # [m/s]
        self.max_vel = 150 / 3.6  # [m/s]
        self.min_vel_increase = 3  # [m/s]

        # Full stop parameters
        self.min_full_stop_time = 2  # [s]
        self.max_full_stop_time = 60  # [s]

        # General parameters
        self.sigma_vel = 0.05  # [m/s]

    def generate_profile(self, n, plot=False, seed=0):
        # Generate profile
        np.random.seed(seed)
        t = np.zeros(n)
        v = np.zeros(n)
        s = np.zeros(n, dtype=np.uint)
        s[0] = 3  # We start with full stop
        for i in range(1, n):
            if s[i - 1] == 0 and v[i - 1] == 0:
                s[i] = 3
            else:
                valid = False
                while not valid:
                    s[i] = np.searchsorted(np.cumsum(self.transition[s[i - 1]]), np.random.rand())
                    valid = True
                    if s[i] == 2 and v[i - 1] > self.max_vel_accelerating:
                        valid = False

            if s[i] == 0:
                # We are braking
                vel_reduction = 0
                while vel_reduction < self.min_vel_reduction:
                    vel_reduction = np.random.gamma(self.brk_shape, self.brk_std * (1 + 2 * v[i] / self.max_vel))
                dec = self.decelerations[np.random.randint(len(self.decelerations))]
                if v[i - 1] - vel_reduction < self.min_vel:
                    vel_reduction = v[i - 1]
                elif np.random.rand() < self.p_full_stop:
                    vel_reduction = v[i - 1]
                t[i] = t[i - 1] + vel_reduction / dec
                v[i] = v[i - 1] - vel_reduction
            elif s[i] == 1:
                # We are cruising
                time = np.random.rand() * (self.max_cruising_time - self.min_cruising_time) + self.min_cruising_time
                vel_diff = (2 * np.random.rand() - 1) * self.max_vel_difference
                t[i] = t[i - 1] + time
                v[i] = v[i - 1] + vel_diff
            elif s[i] == 2:
                # We are accelerating
                vel_increase = 0
                while vel_increase < self.min_vel_increase or vel_increase + v[i - 1] > self.max_vel:
                    vel_increase = np.random.gamma(self.acc_shape, self.acc_std * (3 - 2 * v[i] / self.max_vel))
                acc = self.decelerations[np.random.randint(len(self.decelerations))]
                t[i] = t[i - 1] + vel_increase / acc
                v[i] = v[i - 1] + vel_increase
            else:  # s[i] == 3
                # We are in a full stop
                time = np.random.rand() * (
                    self.max_full_stop_time - self.min_full_stop_time) + self.min_full_stop_time
                t[i] = t[i - 1] + time

        # Generate the time sequences
        tt = np.arange(0, t[-1], self.ts)
        vv = np.zeros_like(tt)
        yy = np.zeros_like(tt, dtype=np.uint)
        # Loop through all events
        iend = 0
        for i in range(1, n):
            istart = iend
            iend = np.searchsorted(tt, t[i])
            if s[i] == 0 or s[i] == 1 or s[i] == 2:
                # Pick profile from dfs
                df = self.dfs[np.random.randint(len(self.dfs))]
                tscale = (tt[istart:iend] - tt[istart]) / (tt[iend - 1] - tt[istart])
                vscale = np.interp(tscale, df['time'], df['vel'])
                vv[istart:iend] = vscale * (v[i - 1] - v[i]) + v[i]
                # vv[istart:iend] = np.interp(tt[istart:iend], tt[[istart, iend-1]], v[[i-1, i]])
            yy[istart:iend] = s[i]
        vv += np.random.normal(0, self.sigma_vel, vv.shape)

        # Plot signal
        if plot:
            plt.subplots(1, 1, figsize=(16, 5))
            plt.vlines(t, 0, np.max(vv) * 3.6, linestyles='dashed', color=[0.5, 0.5, 0.5])
            plt.plot(tt, vv * 3.6, linewidth=3)
            plt.xlim([0, tt[-1]])
            plt.ylim([0, np.max(vv) * 3.6])
            plt.show()

        # Return the result
        return vv, yy


if __name__ == '__main__':
    pg = ProfileGeneration()
    _, _ = pg.generate_profile(100, plot=True)
