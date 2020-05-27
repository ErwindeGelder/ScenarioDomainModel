import collections
import numpy as np


class IDM:
    def __init__(self, a, b, delta, s0, T, amin, tr):
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.T = T
        self.amin = amin
        self.tr = tr

        self.t, self.v, self.x = 0, 0, 0
        self.v0 = 0

        self.times = collections.deque(maxlen=int(self.tr * 200))
        self.accelerations = collections.deque(maxlen=len(self.times))

    def init_simulation(self, x0, v0):
        self.times = collections.deque(maxlen=int(self.tr * 100))
        self.accelerations = collections.deque(maxlen=int(self.tr * 100))
        self.times.append(-1000)
        self.times.append(-0.01)
        self.accelerations.append(0)
        self.accelerations.append(0)
        self.t, self.v, self.x = 0, v0, x0
        self.v0 = v0 * 10

    def step_simulation(self, t, xlead, vlead):
        # Update speed
        a = np.max((self.amin, np.interp(t - self.tr, self.times, self.accelerations)))
        self.v += a * (t - self.t)

        # Update position
        self.x += self.v * (t - self.t)

        # Calculate acceleration based on IDM
        sstar = self.s0 + self.v * self.T + self.v * (self.v - vlead) / (
                    2 * np.sqrt(self.a * self.b))
        self.accelerations.append(self.a * (1 - (self.v / self.v0) ** self.delta -
                                            (sstar / (xlead - self.x)) ** 2))
        self.times.append(t)
        self.t = t

        return self.x, self.v
