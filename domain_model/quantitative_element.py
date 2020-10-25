""" Class QuantitativeElement

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
2020 08 22: Add function to obtain properties from a dictionary.
2020 10 25: Change QuantitativeElement to QuantitativeElement.
"""

from abc import abstractmethod
from .scenario_element import ScenarioElement, _scenario_element_props_from_json


class QuantitativeElement(ScenarioElement):
    """ ScenarioElement that is used for the quantitative classes.

    There are no additional attributes than the one inherited from
    ScenarioElement. This is an abstract class, so it is not possible to
    instantiate objects from this class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the thing.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        ScenarioElement.__init__(self, **kwargs)


def _quantitative_element_props_from_json(json: dict) -> dict:
    return _scenario_element_props_from_json(json)
