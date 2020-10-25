""" Class PhysicalThing

Creation date: 2020 08 19
Author(s): Erwin de Gelder

Modifications:
2020 08 22: Add function to obtain properties from a dictionary.
2020 10 12: From now on, PhysicalThing is not an abstract class.
"""

from typing import Union
from .physical_element_category import PhysicalElementCategory, _physical_element_category_from_json
from .quantitative_element import QuantitativeThing, _quantitative_thing_props_from_json
from .scenario_element import DMObjects, _attributes_from_json, _object_from_json
from .type_checking import check_for_type


class PhysicalThing(QuantitativeThing):
    """ Class for modeling all physical things (both static and dynamic)

    A PhysicalThing is used to quantitatively describe a physical thing.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the physical
            thing.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        category (PhysicalElementCategory): The qualitative counterpart.
        properties(dict): All properties of the physical thing.
    """
    def __init__(self, category: PhysicalElementCategory, properties: dict = None, **kwargs):
        if properties is None:
            properties = dict()
        check_for_type("category", category, PhysicalElementCategory)
        check_for_type("properties", properties, dict)

        QuantitativeThing.__init__(self, **kwargs)
        self.category = category  # type: PhysicalElementCategory
        self.properties = dict() if properties is None else properties

    def get_tags(self) -> dict:
        return self.tags + self.category.get_tags()

    def to_json(self) -> dict:
        thing = QuantitativeThing.to_json(self)
        thing["properties"] = self.properties
        thing["category"] = {"name": self.category.name, "uid": self.category.uid}
        return thing

    def to_json_full(self) -> dict:
        thing = self.to_json()
        thing["category"] = self.category.to_json_full()
        return thing


def _physical_thing_props_from_json(json: dict, attribute_objects: DMObjects,
                                    category: Union[PhysicalElementCategory, bool] = None) -> dict:
    props = _quantitative_thing_props_from_json(json)
    props["properties"] = json["properties"]
    if category is not False:
        props.update(_attributes_from_json(json, attribute_objects,
                                           dict(category=(_physical_element_category_from_json,
                                                          "physical_thing_category")),
                                           category=category))
    return props


def _physical_thing_from_json(json: dict, attribute_objects: DMObjects,
                              category: PhysicalElementCategory = None) -> PhysicalThing:
    return PhysicalThing(**_physical_thing_props_from_json(json, attribute_objects, category))


def physical_thing_from_json(json: dict, attribute_objects: DMObjects = None,
                             category: PhysicalElementCategory = None) -> PhysicalThing:
    """ Get PhysicalThing object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    PhysicalElementCategory needs to be fully defined instead of only the unique
    ID. Alternatively, the PhysicalElementCategory can be passed as optional
    argument. In that case, the PhysicalElementCategory does not need to be
    defined in the JSON code.

    :param json: JSON code of PhysicalThing.
    :param attribute_objects: A structure for storing all objects (optional).
    :param category: If given, it will not be based on the JSON code.
    :return: PhysicalThing object.
    """
    return _object_from_json(json, _physical_thing_from_json, "physical_thing", attribute_objects,
                             category=category)
