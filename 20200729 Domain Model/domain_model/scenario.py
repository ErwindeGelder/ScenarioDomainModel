""" Class Scenario

Creation date: 2018 11 05
Author(s): Erwin de Gelder

To do:
Move falls_into method to ScenarioCategory and rename it to "comprises".

Modifications:
2018 11 22: Make is possible to instantiate a Scenario from JSON code.
2018 12 06: Add functionality to return the derived Tags.
2018 12 06: Make it possible to return full JSON code (incl. attributes' JSON code).
2018 12 06: to_openscenario function added.
2018 12 07: fall_into method for checking if Scenario falls into ScenarioCategory.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 13: Update of terminology.
2019 11 04: Add options to automatically assign unique ids to actor/activities.
"""

from typing import List, Tuple
import fnmatch
import numpy as np
from .thing import Thing
from .static_environment import StaticEnvironment, stat_env_from_json
from .activity import Activity, activity_from_json
from .actor import Actor, actor_from_json
from .tags import tag_from_json
from .scenario_category import ScenarioCategory, derive_actor_tags, create_unique_ids, \
    _check_match_matrix
from .type_checking import check_for_type, check_for_list, check_for_tuple


class Scenario(Thing):
    """ Scenario - either a real-world scenario or a test case.

    A scenario is a quantitative description of the ego vehicle, its activities
    and/or goals, its dynamic environment (consisting of traffic environment and
    conditions) and its static environment. From the perspective of the ego
    vehicle, a scenario contains all relevant events.

    When instantiating the Scenario object, the name, tstart, tend,
    unique id (uid), and tags are passed. To set the activities, actors, and
    acts, use the corresponding methods, i.e., set_activities(), set_actors(),
    and set_acts(), respectively.

    Attributes:
        tstart (float): The start time of the scenario. Part of the time
            intervals of the activities might be before the start time of the
            scenario.
        tend (float): The end time of the scenario. Part of the time interval of
            the activities might be after the end time of the scenario.
        actors (List[Actor]): Actors that are participating in this scenario.
            This list should always include the ego vehicle.
        activities (List[Activity]): Activities that are relevant for this
            scenario.
        acts (List[tuple[Actor, Activity, float]]): The acts describe which
            actors perform which activities.
        name (str): A name that serves as a short description of the scenario.
        uid (int): A unique ID.
        tags (List[Tag]): A list of tags that formally defines the scenario
            category. These tags determine whether a scenario category comprises
            this scenario or not.
    """

    def __init__(self, tstart, tend, static_environment, **kwargs):
        # Check the types of the inputs
        tstart = float(tstart) if isinstance(tstart, int) else tstart
        check_for_type("tstart", tstart, float)
        tend = float(tend) if isinstance(tend, int) else tend
        check_for_type("tend", tend, float)
        check_for_type("static_environment", static_environment, StaticEnvironment)

        # Assign the attributes
        Thing.__init__(self, **kwargs)
        self.time = {"start": tstart, "end": tend}
        self.actors = []      # Type: List[Actor]
        self.activities = []  # Type: List{Activity]
        self.acts = []        # Type: List[tuple(Actor, Activity, float)]
        self.static_environment = static_environment

    def set_activities(self, activities: List[Activity], update_uids: bool = False) -> None:
        """ Set the activities.

        Check whether the activities are correctly defined. Activities should be
        a list with instantiations of Activity.

        :param activities: List of activities that are used for this Scenario.
        :param update_uids: Automatically assign uids if they are similar.
        """
        # Check whether the activities are correctly defined.
        check_for_list("activities", activities, Activity, can_be_none=False)

        # Assign actitivies to an attribute.
        self.activities = activities  # Type: List[Activity]

        # Update the uids of the activities.
        if update_uids:
            create_unique_ids(self.activities)

    def set_actors(self, actors: List[Actor], update_uids: bool = False) -> None:
        """ Set the actors.

        Check whether the actors are correctly defined. Actors should be a list
        with instantiations of Actor.

        :param actors: List of actors that participate in the Scenario.
        :param update_uids: Automatically assign uids if they are similar.
        """
        # Check whether the actors are correctly defined.
        check_for_list("actors", actors, Actor)

        # Assign actors to an attribute.
        self.actors = actors  # Type: List[Actor]

        # Update the uids of the actors.
        if update_uids:
            create_unique_ids(self.actors)

    def set_acts(self, acts_scenario: List[Tuple[Actor, Activity, float]],
                 verbose: bool = True) -> None:
        """ Set the acts

        Check whether the acts are correctly defined. Each act should be a tuple
        with an actor, an activity, and a starting time, i.e., (Actor, Activity,
        float). Acts is a list containing multiple tuples (Actor, Activity,
        float).

        :param acts_scenario: The acts describe which actors perform which
            activities and a certain time. The actors and activities that are
            used in acts should also be passed with the actors and activities
            arguments. If not, a warning will be shown and the corresponding
            actor/activity will be added to the list of actors/activities.
        :param verbose: Set to False if warning should be surpressed.
        """
        check_for_list("acts", acts_scenario, tuple)
        for act in acts_scenario:
            check_for_tuple("act", act, (Actor, Activity, (int, float)))

        # Set the acts.
        self.acts = acts_scenario

        # Check whether the actors/activities defined with the acts are already listed. If not,
        # the corresponding actor/activity will be added and a warning will be shown.
        for actor, activity, _ in self.acts:
            if actor not in self.actors:
                if verbose:
                    print("Actor with name '{:s}' is used with acts but ".format(actor.name) +
                          "not defined in the list of actors.")
                    print("Therefore, the actor is added to the list of actors.")
                self.actors.append(actor)
            if activity not in self.activities:
                if verbose:
                    print("Activity with name '{:s}' is used with acts but".format(activity.name) +
                          " not defined in the list of activities.")
                    print("Therefore, the activity is added to the list of activities.")
                self.activities.append(activity)

    def derived_tags(self) -> dict:
        """ Return all tags, including the tags of the attributes.

        The Scenario has tags, but also its attributes can have tags. More
        specifically, the StaticEnvironmentCategory, each ActorCategory,
        and each ActivityCategory might have tags. A dictionary will be
        returned. Each item of the dictionary contains a list of tags
        corresponding to either the own object (i.e., ScenarioCategory), an
        ActorCategory, or the StaticEnvironment.

        The tags that might be associated with the ActivityCategory are returned
        with the ActorCategory if the corresponding ActorCategory is performing
        that ActivityCategory according to the defined acts.

        :return: List of tags.
        """
        # Instantiate the dictionary.
        tags = {}

        # Provide the tags of the very own object (ScenarioCategory).
        if self.tags:
            tags["{:s}::Scenario".format(self.name)] = self.tags

        # Provide the tags for each Actor.
        tags = derive_actor_tags(self.actors, self.acts, tags=tags)

        # Provide the tags of the StaticEnvironment.
        if self.static_environment.tags or \
                self.static_environment.static_environment_category.tags:
            tags["{:s}::StaticEnvironment".format(self.static_environment.name)] = \
                self.static_environment.tags + \
                self.static_environment.static_environment_category.tags

        # Return the tags.
        return tags

    def falls_into(self, scenario_category: ScenarioCategory) -> bool:
        """ Determine whether the Scenario falls into the ScenarioCategory.

        It is checked whether the passed scenario category comprises this
        Scenario. To determine whether this is the case, only the derived tags
        are used. The derived tags from the ScenarioCategory should be at least
        be present (or subtags of the tags) in the Scenario.

        :param scenario_category: The potential ScenarioCategory it falls into.
        :return: Whether or not the ScenarioCategory comprises the Scenario.
        """
        # Determine the derived tags of the ScenarioCategory.
        sc_tags = scenario_category.derived_tags()  # sc = ScenarioCategory
        s_tags = self.derived_tags()                # s  = Scenario

        # Check for tags directly related to the ScenarioCategory. These tags should be directly
        # present for the scenario.
        if not self._check_tags(sc_tags, s_tags, "ScenarioCategory", "Scenario"):
            return False

        # Check for tags related to the StaticEnvironment.
        if not self._check_tags(sc_tags, s_tags, "StaticEnvironmentCategory", "StaticEnvironment"):
            return False

        # Check for the actors
        sc_actors = fnmatch.filter(sc_tags, "*::ActorCategory")
        s_actors = fnmatch.filter(s_tags, "*::Actor")
        if len(sc_actors) > len(s_actors):  # In this case, there are less actors in the Scenario.
            return False

        # Create a boolean matrix, where the (i,j)-th element is True if the i-th actor of the
        # ScenarioCategory (i.e., ActorCategory) might correspond to the j-th actor of the Scenario.
        match = np.zeros((len(sc_actors), len(s_actors)), dtype=np.bool)
        for i, sc_actor in enumerate(sc_actors):
            for j, s_actor in enumerate(s_actors):
                match[i, j] = (all(any(map(tag.is_supertag_of, s_tags[s_actor]))
                                   for tag in sc_tags[sc_actor]))

        return _check_match_matrix(match)

    @staticmethod
    def _check_tags(sc_tags: dict, s_tags: dict,
                    sc_class: str = "ScenarioCategory", s_class: str = "Scenario") -> bool:
        """ Check for tags of the ScenarioCategory in the Scenario.

        Check for the tags that are related to the ScenarioCategory's attribute
        specified my sc_class in the Scenario's attribute s_class. This can be
        for the ScenarioCategory itself (default, sc_class="ScenarioCategory"
        and s_class="Scenario") or the StaticEnvironment
        (sc_class="StaticEnvironmentCategory" and s_class="StaticEnvironment").

        :param sc_tags: Dictionary of the derived tags of the ScenarioCategory.
        :param s_tags: Dictionary of the derived tags of the Scenario.
        :param sc_class: Specify attribute to be used of the ScenarioCategory.
        :param s_class: Specify attribute to be used of the Scenario.
        :return: Whether the tags of the ScenarioCategory are found in the tags
            of the Scenario.
        """
        sc_keys = fnmatch.filter(sc_tags, "*::{:s}".format(sc_class))
        if sc_keys:  # In this case, there are tags directly related to the ScenarioCategory.
            s_keys = fnmatch.filter(s_tags, "*::{:s}".format(s_class))
            if s_keys:  # There are tags directly related to the Scenario.
                for tag in sc_tags[sc_keys[0]]:
                    if not any(map(tag.is_supertag_of, s_tags[s_keys[0]])):
                        return False  # A tag of the ScenarioCategory is not found in the Scenario.
            else:  # There are no tags at all directly related to the Scenario.
                return False
        return True

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the Scenario to JSON.

        :return: dictionary that can be converted to a json file.
        """
        scenario = Thing.to_json(self)
        scenario["starttime"] = self.time["start"]
        scenario["endtime"] = self.time["end"]
        scenario["duration"] = self.time["end"] - self.time["start"]
        scenario["static_environment"] = {"name": self.static_environment.name,
                                          "uid": self.static_environment.uid}
        scenario["actor"] = [{'name': actor.name, 'uid': actor.uid} for actor in self.actors]
        scenario["activity"] = [{'name': activity.name, 'uid': activity.uid}
                                for activity in self.activities]
        scenario["act"] = [{"actor": actor.uid, "activity": activity.uid, "starttime": starttime}
                           for actor, activity, starttime in self.acts]
        return scenario

    def to_json_full(self) -> dict:
        """ Get full JSON code of object.

        As opposed to the to_json() method, this method can be used to fully
        construct the object. It might be that the to_json() code links to its
        attributes with only a unique id and name. With this information the
        corresponding object can be looked up into the database. This method
        returns all information, which is not meant for the database, but can be
        used instead for describing a scenario without the need of referring to
        the database.

        :return: dictionary that can be converted to a json file.
        """
        scenario = self.to_json()
        scenario["static_environment"] = self.static_environment.to_json_full()
        scenario["activity"] = [activity.to_json_full() for activity in self.activities]
        scenario["actor"] = [actor.to_json_full() for actor in self.actors]
        return scenario


def scenario_from_json(json: dict) -> Scenario:
    """ Get Scenario object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    StaticEnvironment, all Actor, and all Activity need to be defined, instead
    of only a reference to their IDs.

    :param json: JSON code of Scenario.
    :return: Scenario object.
    """
    scenario = Scenario(json["starttime"],
                        json["endtime"],
                        stat_env_from_json(json["static_environment"]),
                        name=json["name"],
                        uid=int(json["id"]),
                        tags=[tag_from_json(tag) for tag in json["tag"]])

    # Create the actor categories (ActorCategory) and actors (Actor).
    actor_cats = []
    actor_cat_uids = []
    actors = []
    for actor in json["actor"]:
        if "id" in actor["actor_category"]:
            # In this case, it is assumed that the JSON code of the ActorCategory is available.
            actors.append(actor_from_json(actor))
            actor_cats.append(actors[-1].actor_category)
            actor_cat_uids.append(actor_cats[-1].uid)
        else:
            # In this case, the ActorCategory is already defined and we need to reuse that one.
            actor_category = actor_cats[actor_cat_uids.index(actor["actor_category"]["uid"])]
            actors.append(actor_from_json(actor, actor_category=actor_category))
    scenario.set_actors(actors)

    # Create the activity categories (ActivityCategory) and activities (Activity).
    activity_cats = []
    activity_cat_uids = []
    activities = []
    for activity in json["activity"]:
        if "id" in activity["activity_category"]:
            # In this case, it is assumed that the JSON code of the ActivityCategory is available.
            activities.append(activity_from_json(activity))
            activity_cats.append(activities[-1].activity_category)
            activity_cat_uids.append(activity_cats[-1].uid)
        else:
            # In this case, the ActivityCategory is already defined and we need to reuse that one.
            act_cat = activity_cats[activity_cat_uids.index(activity["activity_category"]["uid"])]
            activities.append(activity_from_json(activity, activity_category=act_cat))
    scenario.set_activities(activities)

    # Create the acts.
    actor_uids = [actor.uid for actor in actors]
    activity_uids = [activity.uid for activity in activities]
    acts = [(actors[actor_uids.index(act["actor"])],
             activities[activity_uids.index(act["activity"])],
             act["starttime"])
            for act in json["act"]]
    scenario.set_acts(acts)

    # We are done, so we can return the scenario.
    return scenario
