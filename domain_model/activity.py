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
2020 07 31: Change name of DetectedActivity to SetActivity.
2020 08 23: Make Activity a subclass of TimeInterval.
2020 08 24: Add functionality to obtain the values of the state variables (and the derivative).
2020 10 05: Change way of creating object from JSON code.
2020 10 29: Add plot functionality.
"""

from typing import List, Union
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from .activity_category import ActivityCategory, _activity_category_from_json
from .event import Event
from .scenario_element import DMObjects, _object_from_json, _attributes_from_json
from .time_interval import TimeInterval, _time_interval_props_from_json
from .type_checking import check_for_type


class Activity(TimeInterval):
    """ Activity

    An activity specifies the evolution of a state (defined by ActivityCategory)
    over time. The evolution is described by a model (defined by
    ActivityCategory) and parameters.

    Attributes:
        name (str): A name that serves as a short description of the activity.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        start (Event): The starting event.
        end (Event): The end event.
        category(ActivityCategory): The category of the activity
            defines the state and the model.
        parameters(dict): A dictionary of the parameters that quantifies the
            activity.
    """
    def __init__(self, category: ActivityCategory, parameters: dict, **kwargs):
        # Check the types of the inputs
        check_for_type("activity_category", category, ActivityCategory)
        check_for_type("parameters", parameters, dict)

        TimeInterval.__init__(self, **kwargs)
        self.category = category  # type: ActivityCategory
        self.parameters = parameters  # type: dict

    def get_state(self, npoints: int = 100, time: Union[np.ndarray, float, List] = None) \
            -> np.ndarray:
        """ Obtain the state evaluated at given time instances.

        By default, the state is returned for 100 points equally distributed
        over time. To change the number of points, the argument `npoints` can be
        used. Alternatively, if the argument `time` is used, the state is
        evaluated at the provided time instances.

        :param npoints: Number of points for evaluating the state.
        :param time: Time instance(s) at which the model is to be evaluated.
        :return: Numpy array with the state.
        """
        return self.category.model.get_state(self.parameters,
                                             self._get_time(npoints, time))

    def get_state_dot(self, npoints: int = 100, time: Union[np.ndarray, float, List] = None) \
            -> np.ndarray:
        """ Obtain the derivative of a state evaluated at given time instances.

        By default, the state derivative is returned for 100 points equally
        distributed over time. To change the number of points, the argument
        `npoints` can be used. Alternatively, if the argument `time` is used,
        the state is evaluated at the provided time instances.

        :param npoints: Number of points for evaluating the state derivative.
        :param time: Time instance(s) at which the model is to be evaluated.
        :return: Numpy array with the state.
        """
        state_dot = self.category.model.get_state_dot(self.parameters,
                                                      self._get_time(npoints, time))
        duration = self.get_duration()
        if duration is not None:
            return state_dot / duration
        return state_dot

    def _get_time(self, npoints: int = 100, time: Union[np.ndarray, float, List] = None) \
            -> np.ndarray:
        if time is None:
            return np.linspace(0, 1, npoints)

        # Make sure that the time vector is an numpy array.
        if isinstance(time, (float, int)):
            time = np.array([time])
        elif isinstance(time, List):
            time = np.array(time)

        # See if we have enough information to scale the time data. This is only possible if the
        # start and end time of the activity are known.
        tstart = self.get_tstart()
        tend = self.get_tend()
        if tstart is not None and tend is not None:
            return (time - tstart) / (tend - tstart)
        return time

    def plot(self, axes: Axes = None, **kwargs) -> Axes:
        """ Plot the state variable over time.

        :param axes: Optional. If provided, the state will be plotted using the
            provided axes.
        :param kwargs: Optional arguments that are passed to matplotlib's plot
            function.
        :return: The axes that are used for plotting.
        """
        if axes is None:
            axes = plt.axes()
            axes.set_xlabel("Time [s]")
            axes.set_ylabel(self.category.state.value)

        # Show the state.
        time = self._get_time()
        tstart = self.get_tstart()
        tend = self.get_tend()
        if tstart is not None and tend is not None:
            time = time * (tend - tstart) + tstart
        state = self.get_state()
        axes.plot(time, state, **kwargs)

        return axes

    def get_tags(self) -> dict:
        """ Return the list of tags related to this Activity.

        It returns the tags associated to this Activity and the tags associated
        with the ActivityCategory.

        :return: List of tags.
        """
        return self.tags + self.category.get_tags()

    def to_json(self) -> dict:
        activity = TimeInterval.to_json(self)
        activity["category"] = dict(name=self.category.name,
                                    uid=self.category.uid)
        activity["parameters"] = self.parameters
        return activity

    def to_json_full(self) -> dict:
        activity = TimeInterval.to_json_full(self)
        activity["category"] = self.category.to_json_full()
        activity["parameters"] = self.parameters
        return activity


def _activity_props_from_json(json: dict, attribute_objects: DMObjects, start: Event = None,
                              end: Event = None, category: ActivityCategory = None) -> dict:
    props = dict(parameters=json["parameters"])
    props.update(_time_interval_props_from_json(json, attribute_objects, start=start, end=end))
    props.update(_attributes_from_json(json, attribute_objects,
                                       dict(category=(_activity_category_from_json,
                                                      "activity_category")), category=category))
    return props


def _activity_from_json(json: dict, attribute_objects: DMObjects, start: Event = None,
                        end: Event = None, category: ActivityCategory = None) -> Activity:
    return Activity(**_activity_props_from_json(json, attribute_objects, start, end, category))


def activity_from_json(json: dict, attribute_objects: DMObjects = None, start: Event = None,
                       end: Event = None, category: ActivityCategory = None) -> Activity:
    """ Get Activity object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    ActivityCategory needs to be fully defined instead of only the unique ID.
    Alternatively, the ActivityCategory can be passed as optional argument. In
    that case, the ActivityCategory does not need to be defined in the JSON
    code.
    The same applies for the Event that defines the start of the activity and
    the Event that defines the end of the activity.

    :param json: JSON code of Activity.
    :param attribute_objects: A structure for storing all objects (optional).
    :param start: Event that defines the start. If given, it will not be based
        on the JSON code.
    :param end: Event that defines the end. If given, it will not be based on
        the JSON code.
    :param category: If given, it will not be based on the JSON code.
    :return: Activity object.
    """
    return _object_from_json(json, _activity_from_json, "activity", attribute_objects, start=start,
                             end=end, category=category)
