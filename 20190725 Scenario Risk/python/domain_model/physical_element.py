""" Class PhysicalElement

Creation date: 2020 08 19
Author(s): Erwin de Gelder

Modifications:
2020 08 22: Add function to obtain properties from a dictionary.
2020 10 12: From now on, PhysicalThing is not an abstract class.
2020 10 25: Change PhysicalThing to PhysicalElement.
"""

from typing import Union
from .physical_element_category import PhysicalElementCategory, _physical_element_category_from_json
from .quantitative_element import QuantitativeElement, _quantitative_element_props_from_json
from .scenario_element import DMObjects, _attributes_from_json, _object_from_json
from .type_checking import check_for_type


class PhysicalElement(QuantitativeElement):
    """ Class for modeling all physical elements (both static and dynamic)

    A PhysicalElement is used to quantitatively describe an object that exists
    in the three-dimensional space.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the physical
            element.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        category (PhysicalElementCategory): The qualitative counterpart.
        properties(dict): All properties of the physical element.
    """
    def __init__(self, category: PhysicalElementCategory, properties: dict = None, **kwargs):
        if properties is None:
            properties = dict()
        check_for_type("category", category, PhysicalElementCategory)
        check_for_type("properties", properties, dict)

        QuantitativeElement.__init__(self, **kwargs)
        self.category = category  # type: PhysicalElementCategory
        self.properties = dict() if properties is None else properties

    def get_tags(self) -> dict:
        return self.tags + self.category.get_tags()

    def to_json(self) -> dict:
        thing = QuantitativeElement.to_json(self)
        thing["properties"] = self.properties
        thing["category"] = {"name": self.category.name, "uid": self.category.uid}
        return thing

    def to_json_full(self) -> dict:
        thing = self.to_json()
        thing["category"] = self.category.to_json_full()
        return thing


def _physical_element_props_from_json(json: dict, attribute_objects: DMObjects,
                                      category: Union[PhysicalElementCategory, bool] = None) \
        -> dict:
    props = _quantitative_element_props_from_json(json)
    props["properties"] = json["properties"]
    if category is not False:
        props.update(_attributes_from_json(json, attribute_objects,
                                           dict(category=(_physical_element_category_from_json,
                                                          "physical_element_category")),
                                           category=category))
    return props


def _physical_element_from_json(json: dict, attribute_objects: DMObjects,
                                category: PhysicalElementCategory = None) -> PhysicalElement:
    return PhysicalElement(**_physical_element_props_from_json(json, attribute_objects, category))


def physical_element_from_json(json: dict, attribute_objects: DMObjects = None,
                               category: PhysicalElementCategory = None) -> PhysicalElement:
    """ Get PhysicalElement object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    PhysicalElementCategory needs to be fully defined instead of only the unique
    ID. Alternatively, the PhysicalElementCategory can be passed as optional
    argument. In that case, the PhysicalElementCategory does not need to be
    defined in the JSON code.

    :param json: JSON code of PhysicalElement.
    :param attribute_objects: A structure for storing all objects (optional).
    :param category: If given, it will not be based on the JSON code.
    :return: PhysicalElement object.
    """
    return _object_from_json(json, _physical_element_from_json, "physical_element",
                             attribute_objects, category=category)
