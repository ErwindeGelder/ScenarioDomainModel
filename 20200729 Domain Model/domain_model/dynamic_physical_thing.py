""" Class DynamicPhysicalThing

Creation date: 2020 08 19
Author(s): Erwin de Gelder

Modifications:
"""

from typing import List
from .dynamic_physical_thing_category import DynamicPhysicalThingCategory
from .physical_thing import PhysicalThing
from .state import State
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
    def __init__(self, dynamic_physical_thing_category: DynamicPhysicalThingCategory,
                 initial_states: List[State] = None,
                 properties: dict = None, **kwargs):
        check_for_type("dynamic_physical_thing_category", dynamic_physical_thing_category,
                       DynamicPhysicalThingCategory)
        check_for_list("initial_states", initial_states, State)

        PhysicalThing.__init__(self, properties=properties, **kwargs)
        self.category = dynamic_physical_thing_category
        self.initial_states = [] if initial_states is None else initial_states

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
        dynamic_physical_thing = DynamicPhysicalThing.to_json(self)
        dynamic_physical_thing["category"] = {"name": self.category.name, "uid": self.category.uid}
        dynamic_physical_thing["initial_states"] = [initial_state.to_json()
                                                    for initial_state in self.initial_states]
        return dynamic_physical_thing

    def to_json_full(self) -> dict:
        dynamic_physical_thing = self.to_json()
        dynamic_physical_thing["category"] = self.category.to_json_full()
        return dynamic_physical_thing
