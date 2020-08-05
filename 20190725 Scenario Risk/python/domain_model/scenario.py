""" Class Scenario

Creation date: 2018 11 05
Author(s): Erwin de Gelder

To do:
Move falls_into method to ScenarioCategory and rename it to "comprises".

Modifications:
2018 11 22 Make is possible to instantiate a Scenario from JSON code.
2018 12 06: Add functionality to return the derived Tags.
2018 12 06: Make it possible to return full JSON code (incl. attributes' JSON code).
2018 12 06: to_openscenario function added.
2018 12 07: fall_into method for checking if Scenario falls into ScenarioCategory.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 13: Update of terminology.
2019 11 04: Add options to automatically assign unique ids to actor/activities.
2020 03 27: Enable instantiation from json without needing full json code.
2020 07 03: Enable evaluating a state variable of an actor.
2020 07 15: Check for unique actor names and provide functionality to get actor by its name.
2020 07 30: Enable evaluating the derivative of a state variable of an actor.
"""

from typing import List, Tuple, Union
import fnmatch
import numpy as np
from .default_class import Default
from .state import StateVariable
from .static_environment import StaticEnvironment, stat_env_from_json
from .activity import Activity, activity_from_json
from .actor import Actor, actor_from_json
from .tags import tag_from_json
from .scenario_category import ScenarioCategory, derive_actor_tags, create_unique_ids
from .type_checking import check_for_type, check_for_list, check_for_tuple


class Scenario(Default):
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
        Default.__init__(self, **kwargs)
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

    def set_actors(self, actors: List[Actor], update_uids: bool = False,
                   check_names: bool = True) -> None:
        """ Set the actors.

        Check whether the actors are correctly defined. Actors should be a list
        with instantiations of Actor.

        :param actors: List of actors that participate in the Scenario.
        :param update_uids: Automatically assign uids if they are similar.
        :param check_names: Whether to check if the names are unique.
        """
        # Check whether the actors are correctly defined.
        check_for_list("actors", actors, Actor)

        # Check whether the names are unique.
        if check_names:
            names = []
            for actor in actors:
                if actor.name in names:
                    print("WARNING: There are multiple actors with the name '{0}'. This might")
                    print("         result in errors when using .get_actor_by_name.")
                    print("         To eliminate this warning, provide actors with unique names")
                    print("         or set checknames=False in the .set_actors function.")
                names.append(actor.name)

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

    def get_state(self, actor: Actor, state: StateVariable, time: Union[float, List, np.ndarray]) \
            -> Union[None, float, np.ndarray]:
        """ Obtain the values of the state variable at the given time instants.

        :param actor: The actor of which the state variable is to be retrieved.
        :param state: The state variable that is to be retrieved.
        :param time: The time instance(s).
        :return: The value of the state variable at the given time instants.
        """
        return self._get_state(actor, state, time)

    def get_state_dot(self, actor: Actor, state: StateVariable,
                      time: Union[float, List, np.ndarray]) -> Union[None, float, np.ndarray]:
        """ Obtain the derivative of the values of the state variable at the given time instants.

        :param actor: The actor of which the state variable is to be retrieved.
        :param state: The state variable that is to be retrieved.
        :param time: The time instance(s).
        :return: The value of the state variable at the given time instants.
        """
        return self._get_state(actor, state, time, derivative=True)

    def _get_state(self, actor: Actor, state: StateVariable, time: Union[float, List, np.ndarray],
                   derivative=False) -> Union[None, float, np.ndarray]:
        vec_time = self._time2vec(time)
        is_valid = False

        # Loop through the acts.
        for my_actor, my_activity, _ in self.acts:
            # Only continue with right actor and activity.
            if my_actor == actor and my_activity.activity_category.state == state:
                # Check if the time span contains time instances that we want to evaluate.
                mask = np.logical_and(vec_time >= my_activity.tstart,
                                      vec_time <= my_activity.tend)
                if np.any(mask):
                    if not derivative:
                        tmp_values = my_activity.get_state(time=vec_time[mask])
                    else:
                        tmp_values = my_activity.get_state_dot(time=vec_time[mask])
                    if not is_valid:
                        if len(tmp_values.shape) == 1:
                            values = np.ones(len(vec_time)) * np.nan
                        else:
                            values = np.ones((len(vec_time), tmp_values.shape[0])) * np.nan
                        is_valid = True
                    values[mask] = tmp_values.T

        if not is_valid:
            return None
        if isinstance(time, float):
            return values[0]
        return values

    @staticmethod
    def _time2vec(time: Union[float, List, np.ndarray]) -> np.ndarray:
        if isinstance(time, float):
            vec_time = np.array([time])
        elif isinstance(time, List):
            vec_time = np.array(time)
        elif isinstance(time, np.ndarray):
            vec_time = time
        else:
            raise TypeError("<time> needs to be of type <float>, <List>, or <np.ndarray>.")
        return vec_time

    def derived_tags(self) -> dict:
        """ Return all tags, including the tags of the attributes.

        The ScenarioCategory has tags, but also its attributes can have tags.
        More specifically, the StaticEnvironmentCategory, each ActorCategory,
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
                match[i, j] = (all(any(map(tag.is_subtag, s_tags[s_actor]))
                                   for tag in sc_tags[sc_actor]))

        # The matching of the actors need to be done. If a match is found, the corresponding
        # ActorCategory (=row) and Actor (=column) will be removed from the match matrix.
        n_matches = 1  # Number of matches to look for.
        while match.size:
            # If there is at least one ActorCategory left that has no match, a False will be
            # returned.
            if not all(np.any(match, axis=1)):
                return False

            sum_match_actor = np.sum(match, axis=0)
            j = next((j for j in range(match.shape[1]) if sum_match_actor[j] == n_matches), -1)
            if j >= 0:  # We found an Actor with only n corresponding ActorCategories.
                i = next(i for i in range(match.shape[0]) if match[i, j])
            else:  # True for an ActorCategory with only n corresponding Actors.
                sum_match_actor = np.sum(match, axis=1)
                i = next((i for i in range(match.shape[0]) if sum_match_actor[i] == n_matches), -1)
                if i >= 0:  # We found an ActorCategory with only n corresponding Actors.
                    j = next(j for j in range(match.shape[1]) if match[i, j])
                else:
                    # Try again for higher n (number of matches)
                    n_matches = n_matches + 1
                    continue
            match = np.delete(np.delete(match, i, axis=0), j, axis=1)
            n_matches = 1

        return True

    def get_actor_by_name(self, name: str) -> Union[Actor, None]:
        """ Get the actor with the provided name.

        If there is no actor with the given name, None is returned. Note that
        as soon as an actor is found with the given name, this actor is retuned.
        Therefore, this function might fail if there are multiple actors with
        the same name.

        :param name: The name of the actor that is to be returned.
        :return: The actor with the given name.
        """
        for actor in self.actors:
            if actor.name == name:
                return actor
        return None

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
                    if not any(map(tag.is_subtag, s_tags[s_keys[0]])):
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
        scenario = Default.to_json(self)
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


def scenario_from_json(json: dict, static_environment: StaticEnvironment = None,
                       actors: List[Actor] = None, activities: List[Activity] = None) -> Scenario:
    """ Get Scenario object from JSON code

    It is assumed that all the attributes are fully defined. Hence, the
    StaticEnvironment, all Actor, and all Activity need to be defined, instead
    of only a reference to their IDs.
    Alternatively, the actors, the activities, and static environment can be
    passed as arguments.

    :param json: JSON code of Scenario.
    :param static_environment: If given, it will not be based on the JSON code.
    :param actors: If given, it will not be based on the JSON code.
    :param activities: If given, it will not be based on the JSON code.
    :return: Scenario object.
    """
    if static_environment is None:
        static_environment = stat_env_from_json(json["static_environment"])
    scenario = Scenario(json["starttime"],
                        json["endtime"],
                        static_environment,
                        name=json["name"],
                        uid=int(json["id"]),
                        tags=[tag_from_json(tag) for tag in json["tag"]])

    # Create the actor categories (ActorCategory) and actors (Actor).
    if actors is None:
        categories = []
        categories_uid = []
        actors = []
        for actor in json["actor"]:
            if "id" in actor["actor_category"]:
                # In this case, it is assumed that the JSON code of the ActorCategory is available.
                actors.append(actor_from_json(actor))
                categories.append(actors[-1].actor_category)
                categories_uid.append(categories[-1].uid)
            else:
                # In this case, the ActorCategory is already defined and we need to reuse that one.
                actor_category = categories[categories_uid.index(actor["actor_category"]["uid"])]
                actors.append(actor_from_json(actor, actor_category=actor_category))
    scenario.set_actors(actors)

    # Create the activity categories (ActivityCategory) and activities (Activity).
    if activities is None:
        categories = []
        categories_uid = []
        activities = []
        for activity in json["activity"]:
            if "id" in activity["activity_category"]:
                # In this case, it is assumed that the JSON code of the ActivityCategory is
                # available.
                activities.append(activity_from_json(activity))
                categories.append(activities[-1].activity_category)
                categories_uid.append(categories[-1].uid)
            else:
                # In this case, the ActivityCategory is already defined and we need to reuse that
                # one.
                act_cat = categories[categories_uid.index(activity["activity_category"]["uid"])]
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

    return scenario
