""" Class Actor

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 22: Remove example. For example, see example_actor.py.
2018 11 22: Make it possible to instantiate Actor from JSON code.
2018 12 06: Make it possible to return full JSON code (incl. attributes' JSON code).
2019 03 13: Add initial/desired state attributes of Actor.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 05 23: Add goal attribute to actor. This can be used when goal cannot be formulated as a
            desried state.
2019 10 11: Update of terminology.
2019 11 04: Add goals to ego vehicle.
2020 08 19: Make Actor a subclass of DynamicPhysicalThing.
2020 08 22: Add function to obtain properties from a dictionary.
2020 10 05: Change way of creating object from JSON code.
2020 10 12: Make Actor a subclass of PhysicalElement instead of DynamicPhysicalThing.
"""

from typing import List
from .actor_category import ActorCategory, _actor_category_from_json
from .physical_element import PhysicalElement, _physical_element_props_from_json
from .scenario_element import DMObjects, _object_from_json, _attributes_from_json
from .state import State, state_from_json
from .tags import Tag
from .type_checking import check_for_type, check_for_list


class Actor(PhysicalElement):
    """ Actor

    An actor is a physical element that is not static. "Ego vehicle" and
    "Other Road User" are types of actors in a scenario.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        properties(dict): All properties of the physical element.
        initial_states (List[State]): Specifying the initial states.
        desired_states (List[State]): Specifying the goal/objectve of the actor.
        category (ActorCategory): The qualitative counterpart.
    """
    def __init__(self, category: ActorCategory, desired_states: List[State] = None,
                 initial_states: List[State] = None, properties: dict = None, **kwargs):
        # Check the types of the inputs
        check_for_type("actor_category", category, ActorCategory)
        check_for_list("initial_states", desired_states, State)
        check_for_list("desired_states", desired_states, State)

        PhysicalElement.__init__(self, category, properties, **kwargs)
        self.initial_states = [] if initial_states is None else initial_states  # type: List[State]
        self.desired_states = [] if desired_states is None else desired_states  # type: List[State]

    def to_json(self) -> dict:
        actor = PhysicalElement.to_json(self)
        actor["initial_states"] = [initial_state.to_json() for initial_state in self.initial_states]
        actor["desired_states"] = [desired_state.to_json() for desired_state in self.desired_states]
        return actor


def _actor_props_from_json(json: dict, attribute_objects: DMObjects,
                           category: ActorCategory = None) -> dict:
    props = dict(initial_states=[state_from_json(state) for state in json["initial_states"]],
                 desired_states=[state_from_json(state) for state in json["desired_states"]])
    props.update(_physical_element_props_from_json(json, attribute_objects, False))
    props.update(_attributes_from_json(json, attribute_objects,
                                       dict(category=(_actor_category_from_json,
                                                      "actor_category")),
                                       category=category))
    return props


def _actor_from_json(json: dict, attribute_objects: DMObjects, category: ActorCategory = None) \
        -> Actor:
    return Actor(**_actor_props_from_json(json, attribute_objects, category))


def actor_from_json(json: dict, attribute_objects: DMObjects = None,
                    category: ActorCategory = None) -> Actor:
    """ Get Actor object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    ActorCategory needs to be fully defined instead of only the unique ID.
    Alternatively, the ActorCategory can be passed as optional argument. In that
    case, the ActorCategory does not need to be defined in the JSON code.

    :param json: JSON code of Actor.
    :param attribute_objects: A structure for storing all objects (optional).
    :param category: If given, it will not be based on the JSON code.
    :return: Actor object.
    """
    return _object_from_json(json, _actor_from_json, "actor", attribute_objects, category=category)


class EgoVehicle(Actor):
    """ Special actor: Ego vehicle.

    The ego vehicle is similarly defined as an actor. The only difference is
    that it contains the tag "Ego vehicle".

    """
    def __init__(self, category: ActorCategory, initial_states: List[State] = None,
                 desired_states: List[State] = None, properties: dict = None, **kwargs):
        Actor.__init__(self, category, initial_states=initial_states,
                       desired_states=desired_states, properties=properties, **kwargs)
        if Tag.EgoVehicle not in self.tags:
            self.tags += [Tag.EgoVehicle]
