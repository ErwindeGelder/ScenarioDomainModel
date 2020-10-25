""" Class PhysicalElementCategory

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
2020 08 25: Add function to obtain properties from a dictionary.
2020 10 12: From now on, PhysicalThingCategory is not an abstract class.
2020 10 25: Change PhysicalThingCategory to PhysicalElementCategory.
"""

from .qualitative_element import QualitativeElement, _qualitative_element_props_from_json
from .scenario_element import DMObjects, _object_from_json


class PhysicalElementCategory(QualitativeElement):
    """ PhysicalElementCategory: Category of a physical thing

    A PhysicalElementCategory is used to qualitatively describe a physical
    element.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the physical
            element category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        description(str): A string that qualitatively describes this physical
            element.
    """
    def __init__(self, **kwargs):
        QualitativeElement.__init__(self, **kwargs)


def _physical_element_category_props_from_json(json):
    return _qualitative_element_props_from_json(json)


def _physical_element_category_from_json(
        json: dict,
        attribute_objects: DMObjects  # pylint: disable=unused-argument
) -> PhysicalElementCategory:
    return PhysicalElementCategory(**_physical_element_category_props_from_json(json))


def physical_element_category_from_json(json: dict, attribute_objects: DMObjects = None) \
        -> PhysicalElementCategory:
    """ Get PhysicalElementCategory object from JSON code

    It is assumed that the JSON code of the PhysicalElementCategory is created
    using PhysicalElementCategory.to_json().

    :param json: JSON code of PhysicalElement.
    :param attribute_objects: A structure for storing all objects (optional).
    :return: PhysicalElementCategory object.
    """
    return _object_from_json(json, _physical_element_category_from_json,
                             "physical_element_category", attribute_objects)
