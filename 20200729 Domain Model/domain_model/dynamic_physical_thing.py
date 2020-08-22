""" Class DynamicPhysicalThing

Creation date: 2020 08 19
Author(s): Erwin de Gelder

Modifications:
2020 08 22: Add function to obtain properties from a dictionary.
"""

from typing import List
from .dynamic_physical_thing_category import DynamicPhysicalThingCategory, \
    dynamic_physical_thing_category_from_json
from .physical_thing import PhysicalThing, _physical_thing_props_from_json
from .state import State, state_from_json
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
        name (str): A name that serves as a short description of the actor
            category.
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
        """ Return the list of tags related to this DynamicPhysicalThing.

        It returns the tags associated to this DynamicPhysicalThing and the tags
        associated with the DynamicPhysicalThingCategory.

        :return: List of tags.
        """
        tags = self.tags
        tags += self.category.get_tags()
        return tags

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


def _dynamic_physical_thing_props_from_json(json: dict) -> dict:
    props = dict(initial_states=[state_from_json(state) for state in json["initial_states"]])
    props.update(_physical_thing_props_from_json(json))
    return props


def dynamic_physical_thing_from_json(json: dict, category: DynamicPhysicalThingCategory = None) \
        -> DynamicPhysicalThing:
    """ Get DynamicPhysicalThing object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    DynamicPhysicalThingCategory needs to be fully defined instead of only the
    unique ID. Alternatively, the DynamicPhysicalThingCategory can be passed as
    optional argument. In that case, the DynamicPhysicalThingCategory does not
    need to be defined in the JSON code.

    :param json: JSON code of DynamicPhysicalThing.
    :param category: If given, it will not be based on the JSON code.
    :return: DynamicPhysicalThing object.
    """
    if category is None:
        category = dynamic_physical_thing_category_from_json(json["category"])
    return DynamicPhysicalThing(category, **_dynamic_physical_thing_props_from_json(json))
