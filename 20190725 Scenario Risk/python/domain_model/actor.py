"""
Class Actor


Author
------
Erwin de Gelder

Creation
--------
30 Oct 2018

To do
-----

Modifications
-------------
05 Nov 2018: Make code PEP8 compliant.
22 Nov 2018: Remove example. For example, see example_actor.py.
22 Nov 2018: Make it possible to instantiate Actor from JSON code.
06 Dec 2018: Make it possible to return full JSON code (incl. attributes' JSON code).
13 Mar 2019: Add initial/desired state attributes of Actor.
22 May 2019: Make use of type_checking.py to shorten the initialization.
23 May 2019: Add goal attribute to actor. This can be used when goal cannot be formulated as a
             desried state.
11 Oct 2019: Update of terminology.
04 Nov 2019: Add goals to ego vehicle.
23 Mar 2020: Add properties attributes.
"""


from typing import List
from .default_class import Default
from .actor_category import ActorCategory, actor_category_from_json
from .tags import Tag, tag_from_json
from .state import State, state_from_json
from .type_checking import check_for_type, check_for_list


class Actor(Default):
    """ Category of actor

    An actor is an agent in a scenario acting on its own behalf. "Ego vehicle"
    and "Other Road User" are types of actors in a scenario. The actor category
    only describes the actor in qualitative terms.

    Attributes:
        actor_category (ActorCategory): Specifying the category to which the
            actor belongs to.
        initial_states (List[State]): Specifying the initial states.
        desired_states (List[State]): Specifying the goal/objectve of the actor.
        goal (str): A goal that cannot be formulated using the desired state.
        proporties (dict): Properties of the actor.
        name (str): A name that serves as a short description of the qualitative
            actor.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, actor_category: ActorCategory, initial_states: List[State] = None,
                 desired_states: List[State] = None, goal: str = "", properties: dict = None,
                 **kwargs):
        # Check the types of the inputs
        check_for_type("actor_category", actor_category, ActorCategory)
        check_for_list("initial_states", initial_states, State)
        check_for_list("desired_states", desired_states, State)
        check_for_type("goal", goal, str)
        if initial_states is None:
            initial_states = []
        if desired_states is None:
            desired_states = []

        Default.__init__(self, **kwargs)
        self.actor_category = actor_category  # type: ActorCategory
        self.initial_states = initial_states  # type: List[State]
        self.desired_states = desired_states  # type: List[State]
        self.goal = goal                      # type: str
        self.properties = properties          # type: dict

    def get_tags(self) -> dict:
        """ Return the list of tags related to this Actor.

        It returns the tags associated to this Actor and the tags associated
        with the ActorCategory.

        :return: List of tags.
        """
        tags = self.tags
        tags += self.actor_category.get_tags()
        return tags

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of Actor to JSON.

        :return: dictionary that can be converted to a json file.
        """
        actor = Default.to_json(self)
        actor["actor_category"] = {"name": self.actor_category.name,
                                   "uid": self.actor_category.uid}
        actor["initial_states"] = [initial_state.to_json() for initial_state in self.initial_states]
        actor["desired_states"] = [desired_state.to_json() for desired_state in self.desired_states]
        actor["properties"] = self.properties
        return actor

    def to_json_full(self) -> dict:
        """ Get full JSON code of object.

        As opposed to the to_json() method, this method can be used to fully
        construct the object. It might be that the to_json() code links to its
        attributes with only a unique id and name. With this information the
        corresponding object can be looked up into the database. This method
        returns all information, which is not meant for the database, but can be
        used instead for describing a scenario without the need of referring to
        the database.

        :return: dictionary that can be converted to a json file
        """
        actor = self.to_json()
        actor["actor_category"] = self.actor_category.to_json_full()
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
        actor_category = actor_category_from_json(json["actor_category"])
    initial_states = [state_from_json(state) for state in json["initial_states"]]
    derired_states = [state_from_json(state) for state in json["desired_states"]]
    actor = Actor(actor_category,
                  initial_states=initial_states,
                  desired_states=derired_states,
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
