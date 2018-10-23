import numpy as np
import matplotlib.pyplot as plt
from ActivityCategory import ActivityCategory


class Activity:
    """ Activity

    An activity specifies the evolution of a state (defined by ActivityCategory) over time.

    """
    def __init__(self, name, activity_category, duration, parameters):
        self.name = name
        self.activity_category = activity_category  # type: ActivityCategory
        self.tduration = duration
        self.parameters = parameters
        self.tstart = 0
        self.tend = self.tduration

    def plot_state(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        x = np.linspace(self.tstart, self.tend, 100)
        y = self.activity_category.model.get_state(self.parameters, n=len(x))
        ax.plot(x, y)
        ax.set_xlabel("Time")
        ax.set_ylabel(self.activity_category.state)

    def plot_state_dot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        x = np.linspace(self.tstart, self.tend, 100)
        y = self.activity_category.model.get_state_dot(self.parameters, n=len(x)) / self.tduration
        ax.plot(x, y)
        ax.set_xlabel("Time")
        ax.set_ylabel("{:s} dot".format(self.activity_category.state))

    def to_json(self):
        """

        :return: dictionary that can be converted to a json file
        """
        activity = {"name": self.name,
                    "activity_category": self.activity_category.name,
                    "tduration": self.tduration,
                    "parameters": self.parameters,
                    "tstart": self.tstart,
                    "tend": self.tend}
        return activity


class DetectedActivity(Activity):
    def __init__(self, name, activity_category, tstart, duration, parameters):
        Activity.__init__(self, name, activity_category, duration, parameters)
        self.tstart = tstart
        self.tend = self.tstart + self.tduration
