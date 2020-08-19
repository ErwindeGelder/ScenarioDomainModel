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
2020 08 19: Make Actor a subclass of DynamicPhysicalThing
"""

from typing import List
from .actor_category import ActorCategory, actor_category_from_json
from .dynamic_physical_thing import DynamicPhysicalThing
from .state import State, state_from_json
from .tags import Tag, tag_from_json
from .type_checking import check_for_type, check_for_list


class Actor(DynamicPhysicalThing):
    """ Actor

    An actor is a dynamic physical thing that can have an intent. "Ego vehicle"
    and "Other Road User" are types of actors in a scenario.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        properties(dict): All properties of the dynamic physical thing.
        initial_states (List[State]): Specifying the initial states.
        desired_states (List[State]): Specifying the goal/objectve of the actor.
        category (ActorCategory): The qualitative counterpart.
    """
    def __init__(self, actor_category: ActorCategory, desired_states: List[State] = None,
                 initial_states: List[State] = None, properties: dict = None, **kwargs):
        # Check the types of the inputs
        check_for_type("actor_category", actor_category, ActorCategory)
        check_for_list("desired_states", desired_states, State)

        DynamicPhysicalThing.__init__(self, actor_category, initial_states, properties, **kwargs)
        self.desired_states = [] if desired_states is None else desired_states  # type: List[State]

    def to_json(self) -> dict:
        actor = DynamicPhysicalThing.to_json(self)
        actor["desired_states"] = [desired_state.to_json() for desired_state in self.desired_states]
        return actor


def actor_from_json(json: dict, actor_category: ActorCategory = None) -> Actor:
    """ Get Actor object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    ActorCategory needs to be fully defined instead of only the unique ID.
    Alternatively, the ActorCategory can be passed as optional argument. In that
    case, the ActorCategory does not need to be defined in the JSON code.

    :param json: JSON code of Actor.
    :param actor_category: If given, it will not be based on the JSON code.
    :return: Actor object.
    """
    if actor_category is None:
        actor_category = actor_category_from_json(json["category"])
    initial_states = [state_from_json(state) for state in json["initial_states"]]
    derired_states = [state_from_json(state) for state in json["desired_states"]]
    actor = Actor(actor_category,
                  initial_states=initial_states,
                  desired_states=derired_states,
                  properties=json["properties"],
                  name=json["name"],
                  uid=int(json["id"]),
                  tags=[tag_from_json(tag) for tag in json["tag"]])
    return actor


class EgoVehicle(Actor):
    """ Special actor: Ego vehicle.

    The ego vehicle is similarly defined as an actor. The only difference is
    that it contains the tag "Ego vehicle".

    """
    def __init__(self, actor_category: ActorCategory, initial_states: List[State] = None,
                 desired_states: List[State] = None, goal: str = "", **kwargs):
        Actor.__init__(self, actor_category, initial_states=initial_states,
                       desired_states=desired_states, goal=goal, **kwargs)
        self.tags += [Tag.EgoVehicle]
