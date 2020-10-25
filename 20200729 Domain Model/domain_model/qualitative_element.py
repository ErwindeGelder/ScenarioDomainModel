""" Class QualitativeElement

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
2020 08 25: Add function to obtain properties from a dictionary.
2020 10 25: Change name from QualitativeThing to QualitativeElement.
"""

from abc import abstractmethod
from .scenario_element import ScenarioElement, _scenario_element_props_from_json
from .type_checking import check_for_type


class QualitativeElement(ScenarioElement):
    """ ScenarioElement that is used for most qualitative classes.

    Next to the attributes of ScenarioElement, a QualitativeElement also has a
    description that can be used to qualitatively describe the thing. This is an
    abstract class, so it is not possible to instantiate objects from this
    class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        description(str): A string that qualitatively describes this thing.
    """
    @abstractmethod
    def __init__(self, description: str = "", **kwargs):
        # Check the types of the inputs
        check_for_type("description", description, str)

        self.description = description  # type: str
        ScenarioElement.__init__(self, **kwargs)

    def to_json(self) -> dict:
        thing = ScenarioElement.to_json(self)
        thing["description"] = self.description
        return thing


def _qualitative_element_props_from_json(json: dict) -> dict:
    props = dict(description=json["description"])
    props.update(_scenario_element_props_from_json(json))
    return props
