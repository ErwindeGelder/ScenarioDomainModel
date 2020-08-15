""" Class Event

Creation date: 2019 02 28
Author(s): Erwin de Gelder

Modifications:
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 13: Update of terminology.
"""

from .thing import Thing
from .tags import tag_from_json
from .type_checking import check_for_type


class Event(Thing):
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

        Thing.__init__(self, **kwargs)
        self.conditions = conditions  # type: dict

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the an Event to JSON.

        :return: dictionary that can be converted to a json file.
        """
        event = Thing.to_json(self)
        event["conditions"] = self.conditions
        return event


def event_from_json(json: dict) -> Event:
    """ Get Event object from JSON code.

    It is assumed that the JSON code of the Event is created using
    Event.to_json().

    :param json: JSON code of Event.
    :return: Event object.
    """
    event = Event(json["conditions"],
                  name=json["name"],
                  uid=int(json["id"]),
                  tags=[tag_from_json(tag) for tag in json["tag"]])
    return event
