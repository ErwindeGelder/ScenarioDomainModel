"""
Class Activity


Author
------
Erwin de Gelder

Creation
--------
30 Oct 2018

To do
-----

Modifications
-------------

"""


import numpy as np
import matplotlib.pyplot as plt
from default_class import Default
from activity_category import ActivityCategory, StateVariable
from model import Model
import json
from tags import Tag
from typing import List


class Activity(Default):
    """ Activity

    An activity specifies the evolution of a state (defined by ActivityCategory) over time. The evolution is described
    by a model (defined by ActivityCategory) and parameters.

    Attributes:
        name (str): A name that serves as a short description of the activity.
        activity_category(ActivityCategory): The category of the activity defines the state and the model.
        duration(float): The duration of the activity.
        parameters(dict): A dictionary of the parameters that quantifies the activity.
        tstart(float): By default, the starting time tstart is 0.
        tend(float): By default, the end time is the same as the duration.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, activity_category, duration, parameters, tags=None):
        # Check the types of the inputs
        if not isinstance(activity_category, ActivityCategory):
            raise TypeError("Input 'activity_category' should be of type <ActivityCategory> but is of type {0}.".
                            format(type(activity_category)))
        if not isinstance(duration, float):
            raise TypeError("Input 'duration' should be of type <float> but is of type {0}.".format(type(duration)))
        if not isinstance(parameters, dict):
            raise TypeError("Input 'parameters' should be of type <dict> but is of type {0}.".format(type(parameters)))

        Default.__init__(self, name, tags=tags)
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

    def get_tags(self):
        tags = self.tags
        tags += self.activity_category.get_tags()
        return tags

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of Activity to JSON.

        :return: dictionary that can be converted to a json file
        """
        activity = Default.to_json(self)
        activity["activity_category"] = self.activity_category.name
        activity["tduration"] = self.tduration
        activity["parameters"] = self.parameters
        return activity


class DetectedActivity(Activity):
    """ Detected activity

    The only difference between DetectedActivity and Activity is that with DetectedActivity, the starting time (tstart)
    and the end time (tend) can be defined.

    Attributes:
        tstart(float): The starting time of the activity.
        tend(float): The end time of the activity.
    """
    def __init__(self, name, activity_category, tstart, duration, parameters, tags=None):
        Activity.__init__(self, name, activity_category, duration, parameters, tags=tags)
        self.tstart = tstart
        self.tend = self.tstart + self.tduration

    def to_json(self):
        activity = Activity.to_json(self)
        activity['tstart'] = self.tstart
        activity['tend'] = self.tend
        return activity


class TriggeredActivity(Activity):
    """ Triggered activity

    A triggered activity is similarly defined as an activity with, in addition, some starting conditions.

    Attributes:
        conditions(dict): A dictionary with the conditions that trigger the start of this activity. The dictionary
            needs to be defined according to OSCConditionGroup from OpenSCENARIO.
    """
    def __init__(self, name, activity_category, duration, parameters, conditions, tags=None):
        Activity.__init__(self, name, activity_category, duration, parameters, tags=tags)
        self.conditions = conditions

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of Activity to JSON.

        :return: dictionary that can be converted to a json file
        """
        activity = Activity.to_json(self)
        activity['conditions'] = self.conditions
        return activity


if __name__ == "__main__":
    # An example to illustrate how an activity can be instantiated.
    braking = ActivityCategory("braking", Model("Spline3Knots"), StateVariable.LONGITUDINAL_POSITION,
                               tags=[Tag.VEH_LONG_ACT_DRIVING_FORWARD_BRAKING])
    brakingact = DetectedActivity("ego_braking", braking, 9.00, 1.98,
                                  {"xstart": 133, "xend": 168, "a1": 1.56e-2, "b1": -6.27e-2, "c1": 1.04, "d1": 0,
                                   "a2": 3.31e-2, "b2": -8.89e-2, "c2": 1.06, "d2": -2.18e-3})

    # Show the tags that are associated with the activity.
    print("Tags of the activity:")
    for t in brakingact.get_tags():
        print(" - {:s}".format(t))

    # Show the JSON code when exporting the ActivityCategory to JSON
    print()
    print("JSON code for the ActivityCategory:")
    print(json.dumps(braking.to_json(), indent=4))

    # Show the JSON code when this activity is exported to JSON
    print()
    print("JSON code for the Activity:")
    print(json.dumps(brakingact.to_json(), indent=4))
