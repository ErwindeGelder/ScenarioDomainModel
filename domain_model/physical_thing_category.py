""" Class PhysicalThingCategory

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
2020 08 25: Add function to obtain properties from a dictionary.
2020 10 12: From now on, PhysicalThingCategory is not an abstract class.
"""

from .qualitative_thing import QualitativeThing, _qualitative_thing_props_from_json
from .thing import DMObjects, _object_from_json


class PhysicalThingCategory(QualitativeThing):
    """ PhysicalThingCategory: Category of a physical thing

    A PhysicalThingCategory is used to qualitatively describe a physical thing.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        description(str): A string that qualitatively describes this thing.
    """
    def __init__(self, **kwargs):
        QualitativeThing.__init__(self, **kwargs)


def _physical_thing_category_props_from_json(json):
    return _qualitative_thing_props_from_json(json)


def _physical_thing_category_from_json(
        json: dict,
        attribute_objects: DMObjects  # pylint: disable=unused-argument
) -> PhysicalThingCategory:
    return PhysicalThingCategory(**_physical_thing_category_props_from_json(json))


def physical_thing_category_from_json(json: dict, attribute_objects: DMObjects = None) \
        -> PhysicalThingCategory:
    """ Get PhysicalThingCategory object from JSON code

    It is assumed that the JSON code of the PhysicalThingCategory is created
    using PhysicalThingCategory.to_json().

    :param json: JSON code of PhysicalThing.
    :param attribute_objects: A structure for storing all objects (optional).
    :return: PhysicalThingCategory object.
    """
    return _object_from_json(json, _physical_thing_category_from_json,
                             "physical_thing_category", attribute_objects)

