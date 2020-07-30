""" Class Activity

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 07: Change use of models.
2018 11 12: Demonstrate fit method of the different models.
2018 11 20: Remove example. For example, see example_activity.py.
2018 11 22: Make it possible to instantiate Actor from JSON code.
2018 12 06: Make it possible to return full JSON code (incl. attributes' JSON code).
2019 02 28: The start of a triggered activity is at an event.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 11: Update of terminology.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .default_class import Default
from .activity_category import ActivityCategory, activity_category_from_json
from .tags import tag_from_json
from .event import Event
from .type_checking import check_for_type


class Activity(Default):
    """ Activity

    An activity specifies the evolution of a state (defined by ActivityCategory)
    over time. The evolution is described by a model (defined by
    ActivityCategory) and parameters.

    Attributes:
        activity_category(ActivityCategory): The category of the activity
            defines the state and the model.
        duration(float): The duration of the activity.
        parameters(dict): A dictionary of the parameters that quantifies the
            activity.
        tstart(float): By default, the starting time tstart is 0.
        tend(float): By default, the end time is the same as the duration.
        name (str): A name that serves as a short description of the activity.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, activity_category: ActivityCategory, duration: float, parameters: dict,
                 **kwargs):
        # Check the types of the inputs
        check_for_type("activity_category", activity_category, ActivityCategory)
        duration = float(duration) if isinstance(duration, int) else duration
        check_for_type("duration", duration, float)
        check_for_type("parameters", parameters, dict)

        Default.__init__(self, **kwargs)
        self.activity_category = activity_category  # type: ActivityCategory
        self.tduration = duration
        self.parameters = parameters
        self.tstart = 0
        self.tend = self.tduration

    def plot_state(self, axes: Axes = None, **kwargs) -> None:
        """ Plot the state over time.

        The state is plotted over the time interval of the Activity. If no axes
        is provided in the arguments, then an axes is created.

        :param axes: Optional axes for plotting.
        """
        if axes is None:
            _, axes = plt.subplots(1, 1)
        xdata = np.linspace(self.tstart, self.tend, 100)
        ydata = self.activity_category.model.get_state(self.parameters, npoints=len(xdata))
        axes.plot(xdata, ydata, **kwargs)
        axes.set_xlabel("Time")
        axes.set_ylabel(self.activity_category.state.name)

    def plot_state_dot(self, axes: Axes = None, **kwargs) -> None:
        """ Plot the state derivative over time.

        The state derivative is plotted over the time interval of the Activity.
        If no axes is provided in the arguments, then an axes is created.

        :param axes: Optional axes for plotting.
        """
        if axes is None:
            _, axes = plt.subplots(1, 1)
        xdata = np.linspace(self.tstart, self.tend, 100)
        ydata = (self.activity_category.model.get_state_dot(self.parameters, npoints=len(xdata)) /
                 self.tduration)
        axes.plot(xdata, ydata, **kwargs)
        axes.set_xlabel("Time")
        axes.set_ylabel("{:s} dot".format(self.activity_category.state.name))

    def get_tags(self) -> dict:
        """ Return the list of tags related to this Activity.

        It returns the tags associated to this Activity and the tags associated
        with the ActivityCategory.

        :return: List of tags.
        """
        tags = self.tags
        tags += self.activity_category.get_tags()
        return tags

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of Activity to
        JSON.

        :return: dictionary that can be converted to a json file.
        """
        activity = Default.to_json(self)
        activity["activity_category"] = {"name": self.activity_category.name,
                                         "uid": self.activity_category.uid}
        activity["tduration"] = self.tduration
        activity["parameters"] = self.parameters
        activity["type"] = "Activity"
        return activity

    def to_json_full(self) -> dict:
        """ Get full JSON code of object.

        As opposed to the to_json() method, this method can be used to fully
        construct the object. It might be that the to_json() code links to its
        attributes with only a unique id and name. With this information the
        corresponding object can be looked up into the database. This method
        returns all information, which is not meant for the database, but can be
        used instead for describing a scenario without the need of referring to
        the database.

        :return: dictionary that can be converted to a json file.
        """
        activity = self.to_json()
        activity["activity_category"] = self.activity_category.to_json_full()
        return activity


class DetectedActivity(Activity):
    """ Detected activity.

    The only difference between DetectedActivity and Activity is that with
    DetectedActivity, the starting time (tstart) and the end time (tend) can be
    defined.

    Attributes:
        tstart(float): The starting time of the activity.
        tend(float): The end time of the activity.
    """
    def __init__(self, activity_category: ActivityCategory, tstart: float, duration: float,
                 parameters: dict, **kwargs):
        # Check the types of the inputs
        tstart = float(tstart) if isinstance(tstart, int) else tstart
        check_for_type("tstart", tstart, float)

        Activity.__init__(self, activity_category, duration, parameters, **kwargs)
        self.tstart = tstart
        self.tend = self.tstart + self.tduration

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of Activity to
        JSON.

        :return: dictionary that can be converted to a json file.
        """
        activity = Activity.to_json(self)
        activity['tstart'] = self.tstart
        activity['tend'] = self.tend
        activity["type"] = "DetectedActivity"
        return activity


class TriggeredActivity(Activity):
    """ Triggered activity

    A triggered activity is similarly defined as an activity with, in addition,
    an event that triggers (starts) the activity.

    Attributes:
        trigger(Event): An event that starts this activity.
    """
    def __init__(self, activity_category, duration, parameters, trigger, **kwargs):
        # Check the types of the inputs
        check_for_type("trigger", trigger, Event)

        Activity.__init__(self, activity_category, duration, parameters, **kwargs)
        self.trigger = trigger

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of Activity to
        JSON.

        :return: dictionary that can be converted to a json file.
        """
        activity = Activity.to_json(self)
        activity["trigger"] = self.trigger.to_json()
        activity["type"] = "TriggeredActivity"
        return activity


def activity_from_json(json: dict, activity_category: ActivityCategory = None) \
        -> Activity:
    """ Get Activity object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    ActivityCategory needs to be fully defined instead of only the unique ID.
    Alternatively, the ActivityCategory can be passed as optional argument. In
    that case, the ActivityCategory does not need to be defined in the JSON
    code.

    :param json: JSON code of Activity.
    :param activity_category: If given, it will not be based on the JSON code.
    :return: Activity object.
    """
    if activity_category is None:
        activity_category = activity_category_from_json(json["activity_category"])
    arguments = {"activity_category": activity_category,
                 "duration": json["tduration"],
                 "parameters": json["parameters"],
                 "name": json["name"],
                 "uid": int(json["id"]),
                 "tags": [tag_from_json(tag) for tag in json["tag"]]}

    # Instantiate the Activity object. It can be a regular Activity, a DetectedActivity or a
    # TriggeredActivity
    if json["type"] == "Activity":
        return Activity(**arguments)
    if json["type"] == "DetectedActivity":
        arguments["tstart"] = json["tstart"]
        return DetectedActivity(**arguments)
    if json["type"] == "TriggeredActivity":
        arguments["conditions"] = json["conditions"]
        return TriggeredActivity(**arguments)
    raise ValueError("Type of activity '{:s}' is not valid.".format(json["type"]))
