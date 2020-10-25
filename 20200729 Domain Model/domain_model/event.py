""" Class Event

Creation date: 2019 02 28
Author(s): Erwin de Gelder

Modifications:
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 13: Update of terminology.
2020 08 22: Change how an event is created from json code. Result is the same.
2020 08 24: Change superclass from ScenarioElement to QuantitativeElement.
2020 10 05: Change way of getting properties of the time interval.
"""

from .quantitative_element import QuantitativeElement, _quantitative_element_props_from_json
from .scenario_element import DMObjects, _object_from_json
from .type_checking import check_for_type


class Event(QuantitativeElement):
    """ Event

    An event refers to a time instant at which a notable change happens. This
    can be a sudden change in a certain state or a variable satisfying a certain
    condition (e.g., distance between two vehicles less than 30 meters). A few
    examples:
     - A vehicle enters a tunnel.
     - Simulation time reaches 10 seconds.
     - The distance of the ego vehicle towards the zebra crossing is less than
       10 meters.

    Attributes:
        conditions (dict): A dictionary describing the conditions of the event.
            The dictionary needs to be defined according to OSCConditionGroup
            from OpenSCENARIO.
        name (str): A name that serves as a short description of the static
            environment category.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, conditions: dict, **kwargs):
        # Check the types of the inputs.
        check_for_type("conditions", conditions, dict)

        QuantitativeElement.__init__(self, **kwargs)
        self.conditions = conditions  # type: dict

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the an Event to JSON.

        :return: dictionary that can be converted to a json file.
        """
        event = QuantitativeElement.to_json(self)
        event["conditions"] = self.conditions
        return event


def _event_props_from_json(json: dict) -> dict:
    props = dict(conditions=json["conditions"])
    props.update(_quantitative_element_props_from_json(json))
    return props


def _event_from_json(
        json: dict,
        attribute_objects: DMObjects  # pylint: disable=unused-argument
) -> Event:
    return Event(**_event_props_from_json(json))


def event_from_json(json: dict, attribute_objects: DMObjects = None) -> Event:
    """ Get Event object from JSON code.

    It is assumed that the JSON code of the Event is created using
    Event.to_json().

    :param json: JSON code of Event.
    :param attribute_objects: A structure for storing all objects (optional).
    :return: Event object.
    """
    return _object_from_json(json, _event_from_json, "event", attribute_objects)
