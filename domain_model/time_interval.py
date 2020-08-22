""" Class TimeInterval

Creation date: 2020 08 22
Author(s): Erwin de Gelder

Modifications:
"""

from abc import abstractmethod
from typing import Union
from .event import Event
from .quantitative_thing import QuantitativeThing
from .type_checking import check_for_type


class TimeInterval(QuantitativeThing):
    """ Thing that is used for the quantitative classes.

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

        QuantitativeThing.__init__(self, **kwargs)
        self.start = start
        self.end = end
