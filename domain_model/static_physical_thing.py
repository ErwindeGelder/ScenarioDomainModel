""" Class StaticPhysicalThing

Creation date: 2020 08 24
Author(s): Erwin de Gelder

Modifications:
2020 10 05: Change way of creating object from JSON code.
"""

from .static_physical_thing_category import StaticPhysicalThingCategory, \
    _static_physical_thing_category_from_json
from .physical_thing import PhysicalThing, _physical_thing_props_from_json
from .thing import DMObjects, _object_from_json, _attributes_from_json
from .type_checking import check_for_type


class StaticPhysicalThing(PhysicalThing):
    """ Class for modeling all static physical things

    A StaticPhysicalThing is used to quantitatively describe a static physical
    thing. A static thing is "static" if it does not change during the scenario.

    As with most quantitative classes, a StaticPhysicalThing contains its
    qualitative counterpart, the StaticPhysicalThingCategory.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the dynamic
            physical thing.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        properties(dict): All properties of the dynamic physical thing.
        initial_states (List[State]): Specifying the initial states.
        category (DynamicPhysicalThingCategory): The qualitative counterpart.
    """

    def __init__(self, category: StaticPhysicalThingCategory, properties: dict = None, **kwargs):
        check_for_type("static_physical_thing_category", category, StaticPhysicalThingCategory)

        PhysicalThing.__init__(self, properties=properties, **kwargs)
        self.category = category  # type: StaticPhysicalThingCategory

    def get_tags(self) -> dict:
        return self.tags + self.category.get_tags()

    def to_json(self) -> dict:
        static_physical_thing = PhysicalThing.to_json(self)
        static_physical_thing["category"] = {"name": self.category.name, "uid": self.category.uid}
        return static_physical_thing

    def to_json_full(self) -> dict:
        static_physical_thing = self.to_json()
        static_physical_thing["category"] = self.category.to_json_full()
        return static_physical_thing


def _static_physical_thing_props_from_json(json: dict, attribute_objects: DMObjects,
                                           category: StaticPhysicalThingCategory = None) -> dict:
    props = _physical_thing_props_from_json(json)
    props.update(_attributes_from_json(json, attribute_objects,
                                       dict(category=(_static_physical_thing_category_from_json,
                                                      "static_physical_thing_category")),
                                       category=category))
    return props


def _static_physical_thing_from_json(json: dict, attribute_objects: DMObjects,
                                     category: StaticPhysicalThingCategory = None) \
        -> StaticPhysicalThing:
    return StaticPhysicalThing(**_static_physical_thing_props_from_json(json, attribute_objects,
                                                                        category))


def static_physical_thing_from_json(json: dict, attribute_objects: DMObjects = None,
                                    category: StaticPhysicalThingCategory = None) \
        -> StaticPhysicalThing:
    """ Get StaticPhysicalThing object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    StaticPhysicalThingCategory needs to be fully defined instead of only the
    unique ID. Alternatively, the StaticPhysicalThingCategory can be passed as
    optional argument. In that case, the StaticPhysicalThingCategory does not
    need to be defined in the JSON code.

    :param json: JSON code of StaticPhysicalThing.
    :param attribute_objects: A structure for storing all objects (optional).
    :param category: If given, it will not be based on the JSON code.
    :return: DynamicPhysicalThing object.
    """
    return _object_from_json(json, _static_physical_thing_from_json, "static_physical_thing",
                             attribute_objects, category=category)
