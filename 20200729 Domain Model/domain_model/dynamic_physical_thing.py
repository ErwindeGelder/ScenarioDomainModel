""" Class DynamicPhysicalThing

Creation date: 2020 08 19
Author(s): Erwin de Gelder

Modifications:
2020 08 22: Add function to obtain properties from a dictionary.
2020 10 05: Change way of creating object from JSON code.
"""

from typing import List, Union
from .dynamic_physical_thing_category import DynamicPhysicalThingCategory, \
    _dynamic_physical_thing_category_from_json
from .physical_thing import PhysicalThing, _physical_thing_props_from_json
from .state import State, state_from_json
from .thing import DMObjects, _object_from_json, _attributes_from_json
from .type_checking import check_for_type, check_for_list


class DynamicPhysicalThing(PhysicalThing):
    """ Class for modeling all dynamic physical things (among which, actors)

    A DynamicPhysicalThing is used to quantitatively describe a dynamic physical
    thing. A dynamic thing is "dynamic" if it changes during the scenario. To
    model this change, activities are used to quantitatively describe the change
    in the applicable state variables. The DynamicPhysicalThing, however,
    describes the initial state variables.

    As with most quantitative classes, a DynamicPhysicalThing contains its
    qualitative counterpart, the DynamicPhysicalThingCategory.

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

    def __init__(self, category: DynamicPhysicalThingCategory, initial_states: List[State] = None,
                 properties: dict = None, **kwargs):
        check_for_type("dynamic_physical_thing_category", category,
                       DynamicPhysicalThingCategory)
        check_for_list("initial_states", initial_states, State)

        PhysicalThing.__init__(self, properties=properties, **kwargs)
        self.category = category  # type: DynamicPhysicalThingCategory
        self.initial_states = [] if initial_states is None else initial_states  # type: List[State]

    def get_tags(self) -> dict:
        return self.tags + self.category.get_tags()

    def to_json(self) -> dict:
        dynamic_physical_thing = PhysicalThing.to_json(self)
        dynamic_physical_thing["category"] = {"name": self.category.name, "uid": self.category.uid}
        dynamic_physical_thing["initial_states"] = [initial_state.to_json()
                                                    for initial_state in self.initial_states]
        return dynamic_physical_thing

    def to_json_full(self) -> dict:
        dynamic_physical_thing = self.to_json()
        dynamic_physical_thing["category"] = self.category.to_json_full()
        return dynamic_physical_thing


def _dynamic_physical_thing_props_from_json(
        json: dict, attribute_objects: DMObjects,
        category: Union[bool, DynamicPhysicalThingCategory] = None) -> dict:
    props = dict(initial_states=[state_from_json(state) for state in json["initial_states"]])
    props.update(_physical_thing_props_from_json(json))
    if category is not False:
        props.update(_attributes_from_json(
            json, attribute_objects, dict(category=(_dynamic_physical_thing_category_from_json,
                                                    "dynamic_physical_thing_category")),
            category=category))
    return props


def _dynamic_physical_thing_from_json(json: dict, attribute_objects: DMObjects,
                                      category: DynamicPhysicalThingCategory = None) \
        -> DynamicPhysicalThing:
    return DynamicPhysicalThing(**_dynamic_physical_thing_props_from_json(json, attribute_objects,
                                                                          category))


def dynamic_physical_thing_from_json(json: dict, attribute_objects: DMObjects = None,
                                     category: DynamicPhysicalThingCategory = None) \
        -> DynamicPhysicalThing:
    """ Get DynamicPhysicalThing object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    DynamicPhysicalThingCategory needs to be fully defined instead of only the
    unique ID. Alternatively, the DynamicPhysicalThingCategory can be passed as
    optional argument. In that case, the DynamicPhysicalThingCategory does not
    need to be defined in the JSON code.

    :param json: JSON code of DynamicPhysicalThing.
    :param attribute_objects: A structure for storing all objects (optional).
    :param category: If given, it will not be based on the JSON code.
    :return: DynamicPhysicalThing object.
    """
    return _object_from_json(json, _dynamic_physical_thing_from_json, "dynamic_physical_thing",
                             attribute_objects, category=category)
