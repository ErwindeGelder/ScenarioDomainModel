import numpy as np
import matplotlib.pyplot as plt
from QualitativeActivity import QualitativeActivity


class Activity:
    def __init__(self, qualitative_activity, duration, parameters):
        self.qualitative_activity = qualitative_activity  # type: QualitativeActivity
        self.tduration = duration
        self.parameters = parameters
        self.tstart = 0
        self.tend = self.tduration

    def plot_state(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        x = np.linspace(self.tstart, self.tend, 100)
        y = self.qualitative_activity.model.get_state(self.parameters, n=len(x))
        ax.plot(x, y)
        ax.set_xlabel("Time")
        ax.set_ylabel(self.qualitative_activity.state)

    def plot_state_dot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        x = np.linspace(self.tstart, self.tend, 100)
        y = self.qualitative_activity.model.get_state_dot(self.parameters, n=len(x)) / self.tduration
        ax.plot(x, y)
        ax.set_xlabel("Time")
        ax.set_ylabel("{:s} dot".format(self.qualitative_activity.state))


class DetectedActivity(Activity):
    def __init__(self, qualitative_activity, tstart, duration, parameters):
        Activity.__init__(self, qualitative_activity, duration, parameters)
        self.tstart = tstart
        self.tend = self.tstart + self.tduration
