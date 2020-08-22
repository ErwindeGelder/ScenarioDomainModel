""" Class QuantitativeThing

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
2020 08 22: Add function to obtain properties from a dictionary.
"""

from abc import abstractmethod
from .thing import Thing, _thing_props_from_json


class QuantitativeThing(Thing):
    """ Thing that is used for the quantitative classes.

    There are no additional attributes than the one inherited from Thing. This
    is an abstract class, so it is not possible to instantiate objects from this
    class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        Thing.__init__(self, **kwargs)


def _quantitative_thing_props_from_json(json: dict) -> dict:
    return _thing_props_from_json(json)
