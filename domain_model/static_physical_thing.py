""" Class StaticPhysicalThing

Creation date: 2020 08 24
Author(s): Erwin de Gelder

Modifications:
"""

from typing import List
from .static_physical_thing_category import StaticPhysicalThingCategory, \
    static_physical_thing_category_from_json
from .physical_thing import PhysicalThing, _physical_thing_props_from_json
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
        """ Return the list of tags related to this StaticPhysicalThing.

        It returns the tags associated to this StaticPhysicalThing and the tags
        associated with the StaticPhysicalThingCategory.

        :return: List of tags.
        """
        return self.tags + self.category.get_tags()

    def to_json(self) -> dict:
        dynamic_physical_thing = PhysicalThing.to_json(self)
        dynamic_physical_thing["category"] = {"name": self.category.name, "uid": self.category.uid}
        return dynamic_physical_thing

    def to_json_full(self) -> dict:
        dynamic_physical_thing = self.to_json()
        dynamic_physical_thing["category"] = self.category.to_json_full()
        return dynamic_physical_thing


def _static_physical_thing_props_from_json(json: dict) -> dict:
    return _physical_thing_props_from_json(json)


def static_physical_thing_from_json(json: dict, category: StaticPhysicalThingCategory = None) \
        -> StaticPhysicalThing:
    """ Get StaticPhysicalThing object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    StaticPhysicalThingCategory needs to be fully defined instead of only the
    unique ID. Alternatively, the StaticPhysicalThingCategory can be passed as
    optional argument. In that case, the StaticPhysicalThingCategory does not
    need to be defined in the JSON code.

    :param json: JSON code of StaticPhysicalThing.
    :param category: If given, it will not be based on the JSON code.
    :return: DynamicPhysicalThing object.
    """
    if category is None:
        category = static_physical_thing_category_from_json(json["category"])
    return StaticPhysicalThing(category, **_static_physical_thing_props_from_json(json))
