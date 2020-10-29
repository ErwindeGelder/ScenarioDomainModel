""" Class TimeInterval

Creation date: 2020 08 22
Author(s): Erwin de Gelder

Modifications:
2020 08 24: Add functionality to obtain the start time, the end time, and the duration.
2020 10 05: Change way of getting properties of the time interval.
"""

from abc import abstractmethod
from typing import Union
from .event import Event, _event_from_json
from .quantitative_element import QuantitativeElement, _quantitative_element_props_from_json
from .scenario_element import DMObjects, _attributes_from_json
from .type_checking import check_for_type


class TimeInterval(QuantitativeElement):
    """ ScenarioElement that is used for the quantitative classes.

    A time interval has a start and an end. Both the start and the end are
    defined by an Event. To initialize the time interval, the time can be passed
    instead of an Event. In that case, an Event will be instantiated with the
    provided time as a property. This is an abstract class, so it is not
    possible to instantiate objects from this class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        start (Event): The starting event.
        end (Event): The end event.
    """
    @abstractmethod
    def __init__(self, start: Union[float, Event], end: Union[float, Event], **kwargs):
        check_for_type("start", start, (Event, float, int))
        check_for_type("end", end, (Event, float, int))
        if isinstance(start, (float, int)):
            start = Event(conditions=dict(time=start))
        if isinstance(end, (float, int)):
            end = Event(conditions=dict(time=end))

        QuantitativeElement.__init__(self, **kwargs)
        self.start = start
        self.end = end

    def to_json(self) -> dict:
        time_interval = QuantitativeElement.to_json(self)
        time_interval["start"] = dict(uid=self.start.uid)
        time_interval["end"] = dict(uid=self.end.uid)
        return time_interval

    def to_json_full(self) -> dict:
        time_interval = self.to_json()
        time_interval["start"] = self.start.to_json_full()
        time_interval["end"] = self.end.to_json_full()
        return time_interval

    def get_tstart(self) -> Union[float, None]:
        """ Obtain the start time (if it is available).

        :return: The start time - if it is available. Otherwise, None.
        """
        try:
            return self.start.conditions["time"]
        except KeyError:
            return None

    def get_tend(self) -> Union[float, None]:
        """ Obtain the end time (if it is available).

        :return: The end time - if it is available. Otherwise, None.
        """
        try:
            return self.end.conditions["time"]
        except KeyError:
            return None

    def get_duration(self) -> Union[float, None]:
        """ Obtain the duration (if it is available).

        :return: The duration - if it is available. Otherwise, None.
        """
        try:
            return self.end.conditions["time"] - self.start.conditions["time"]
        except KeyError:
            return None


def _time_interval_props_from_json(json: dict, attribute_objects: DMObjects, start: Event = None,
                                   end: Event = None) -> dict:
    props = _quantitative_element_props_from_json(json)
    props.update(_attributes_from_json(json, attribute_objects,
                                       dict(start=(_event_from_json, "event"),
                                            end=(_event_from_json, "event")), start=start, end=end))
    return props
