""" Class ScenarioCategory

Creation date: 2018 10 30
Author(s): Erwin de Gelder

To do:
Add "comprises" method based on the "fall_into" method that is defined for Scenario.

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 07: Change use of models.
2018 11 22: Enable instantiation using JSON code.
2018 11 29: Add functionality to return the derived Tags.
2018 12 06: Make it possible to return full JSON code (incl. attributes' JSON code).
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 11: Update of terminology.
2019 11 04: Add options to automatically assign unique ids to actor/activities.
2020 07 30: Update conversion of scenario category to a string.
2020 07 31: Add the includes method.
2020 08 15: Remove static_environment and add static_physical_things.
2020 08 16: Add dynamic_physical_thing_categories.
"""

from __future__ import annotations
from typing import List, Tuple
import fnmatch
import numpy as np
from .activity_category import ActivityCategory, activity_category_from_json
from .actor import Actor
from .actor_category import ActorCategory, actor_category_from_json
from .dynamic_physical_thing import DynamicPhysicalThing
from .dynamic_physical_thing_category import DynamicPhysicalThingCategory, \
    dynamic_physical_thing_category_from_json
from .qualitative_thing import QualitativeThing
from .static_physical_thing_category import StaticPhysicalThingCategory, \
    static_physical_thing_category_from_json
from .tags import tag_from_json
from .type_checking import check_for_type, check_for_list, check_for_tuple


class ScenarioCategory(QualitativeThing):
    """ ScenarioCategory - A qualitative description

    Although a scenario is a quantitative description, there also exists a
    qualitative description of a scenario. We refer to the qualitative
    description of a scenario as a scenario category. The qualitative
    description can be regarded as an abstraction of the quantitative scenario.

    Scenario categories comprise scenarios. A scenario category may comprise
    multiple scenarios. On the other hand, multiple scenario categories may
    comprise the same scenario.

    A scenario category can include another scenario class.

    When instantiating the ScenarioCategory object, the name, description,
    image, unique id (uid), and tags are passed. To set the static physical
    things, activities, actors, and acts, use the corresponding methods, i.e.,
    set_static_physical_things(), set_activities(), set_actors(), and
    set_acts(), respectively.

    Attributes:
        description (str): A description of the scenario class. The objective of
            the description is to make the scenario class human interpretable.
        image (str): Path to image that schematically shows the class.
        static_physical_things (List[StaticEnvironmentCategory]): Static
            environment of the Scenario.
        activity_categories (List[ActivityCategory]): List of activities that
            are used for this ScenarioCategory.
        dynamic_physical_thing_categories
            (List[DynamicPhysicalThingCategories]): List of dynamic physical
            things that participate in the Scenario.
        actor_categories (List[ActorCategory]): List of actors that participate
            in the Scenario.
        acts (List[Tuple[ActorCategory, ActivityCategory]]): The acts describe
            which actors perform which activities.
        name (str): A name that serves as a short description of the scenario
            category.
        uid (int): A unique ID.
        tags (List[Tag]): A list of tags that formally defines the scenario
            category. These tags determine whether scenarios fall into this
            scenario category or not.
    """
    def __init__(self, image: str, description: str = "", **kwargs):
        # Check the types of the inputs
        check_for_type("image", image, str)

        # Assign the attributes
        QualitativeThing.__init__(self, description=description, **kwargs)
        self.image = image
        self.static_physical_things = []  # Type: List[StaticPhysicalThingCategory]
        self.activities = []  # Type: List[ActivityCategory]
        self.dynamic_physical_things = []  # Type: List[DynamicPhysicalThingCategories]
        self.actors = []  # Type: List[ActorCategory]
        self.acts = []  # Type: List[Tuple[ActorCategory, ActivityCategory]]

        # Some parameters
        # Maximum number of characters that are used when printing the general description
        self.maxprintlength = 80

    def set_static_physical_things(self, static_physical_things: List[StaticPhysicalThingCategory],
                                   update_uids: bool = False) -> None:
        """ Set the static physical things

        Check whether the physical things are correctly defined.

        :param static_physical_things: List of static physical thing categories
            that define the static environment qualitatively.
        :param update_uids: Automatically assign uids if they are similar.
        """
        # Check whether the static physical things are correctly defined.
        check_for_list("static_physical_things", static_physical_things,
                       StaticPhysicalThingCategory, can_be_none=False)

        # Assign static physical thing categories to an attribute.
        self.static_physical_things = static_physical_things

        # Update the uids of the static physical thing categories.
        if update_uids:
            create_unique_ids(self.static_physical_things)

    def set_activities(self, activity_categories: List[ActivityCategory],
                       update_uids: bool = False) -> None:
        """ Set the activities

        Check whether the activities are correctly defined. Activities should be
        a list with instantiations of ActivityCategory.

        :param activity_categories: List of activities that are used for this
            ScenarioCategory.
        :param update_uids: Automatically assign uids if they are similar.
        """
        # Check whether the activities are correctly defined.
        check_for_list("activities", activity_categories, ActivityCategory, can_be_none=False)

        # Assign activity categories to an attribute.
        self.activities = activity_categories  # Type: List[ActivityCategory]

        # Update the uids of the activity categories.
        if update_uids:
            create_unique_ids(self.activities)

    def set_dynamic_physical_things(self,
                                    dynamic_physical_things: List[DynamicPhysicalThingCategory],
                                    update_uids: bool = False) -> None:
        """ Set the dynamic physical things

        Check whether the dynamic physical things are correctly defined. Dynamic
        physical things should be a list with instantiations of
        DynamicPhysicalThingCategory.

        :param dynamic_physical_things: List of dynamic physical things that
            participate in the Scenario.
        :param update_uids: Automatically assign uids if they are similar.
        """
        # Check whether the actors are correctly defined.
        check_for_list("dynamic_physical_things", dynamic_physical_things,
                       DynamicPhysicalThingCategory, can_be_none=False)

        # Assign actor categories to an attribute.
        self.dynamic_physical_things = dynamic_physical_things

        # Update the uids of the activity categories.
        if update_uids:
            create_unique_ids(self.dynamic_physical_things)

    def set_actors(self, actor_categories: List[ActorCategory], update_uids: bool = False) -> None:
        """ Set the actors

        Check whether the actors are correctly defined. Actors should be a list
        with instantiations of ActorCategory.

        :param actor_categories: List of actors that participate in the
            Scenario.
        :param update_uids: Automatically assign uids if they are similar.
        """
        # Check whether the actors are correctly defined.
        check_for_list("actors", actor_categories, ActorCategory, can_be_none=False)

        # Assign actor categories to an attribute.
        self.actors = actor_categories  # Type: List[ActorCategory]

        # Update the uids of the activity categories.
        if update_uids:
            create_unique_ids(self.actors)

    def set_acts(self, acts_scenario_category: List[Tuple[DynamicPhysicalThingCategory,
                                                          ActivityCategory]],
                 verbose: bool = True) -> None:
        """ Set the acts

        Check whether the acts are correctly defined. Each act should be a tuple
        with a dynamic physical thing and an activity category, i.e.,
        (DynamicPhysicalThingCategory, ActivityCategory). Acts is a list
        containing multiple tuples
        (DynamicPhysicalThingCategory, ActivityCategory).

        :param acts_scenario_category: The acts describe which dynamic physical
            things perform which activities. The dynamic physical things and
            activities that are used in acts should also be passed with the
            dynamic physical things and activities arguments. If not, a warning
            will be shown and the corresponding dynamic physical thing/activity
            will be added to the list of dynamic physical thing/activities.
        :param verbose: Set to False if warning should be surpressed.
        """
        check_for_list("acts", acts_scenario_category, tuple)
        for act in acts_scenario_category:
            check_for_tuple("act", act, (DynamicPhysicalThingCategory, ActivityCategory))

        # Set the acts.
        self.acts = acts_scenario_category

        # Check whether the actors/activities defined with the acts are already listed. If not,
        # the corresponding actor/activity will be added and a warning will be shown.
        for thing, activity in self.acts:
            if thing not in self.actors + self.dynamic_physical_things:
                if verbose:
                    print("Actor/dynamic physical thing with name '{:s}' ".format(thing.name) +
                          "is used with acts but not defined in the list of actors.")
                    print("Therefore, the actor is added to the list of actors.")
                if isinstance(thing, ActorCategory):
                    self.actors.append(thing)
                else:
                    self.dynamic_physical_things.append(thing)
            if activity not in self.activities:
                if verbose:
                    print("Activity with name '{:s}' is used with acts but".format(activity.name) +
                          " not defined in the list of activities.")
                    print("Therefore, the activity is added to the list of activities.")
                self.activities.append(activity)

    def derived_tags(self) -> dict:
        """ Return all tags, including the tags of the attributes.

        The ScenarioCategory has tags, but also its attributes can have tags.
        More specifically, the each StaticPhysicalThingCategory, ActorCategory,
        DynamicPhysicalThingCategory, and ActivityCategory might have tags. A
        dictionary will be returned. Each item of the dictionary contains a list
        of tags corresponding to either the own object (i.e., ScenarioCategory),
        an StaticPhysicalThingCategory, ActorCategory, or
        DynamicPhysicalThingCategory.

        The tags that might be associated with the ActivityCategory are returned
        with the ActorCategory if the corresponding ActorCategory is performing
        that ActivityCategory according to the defined acts.

        :return: List of tags.
        """
        # Instantiate the dictionary.
        tags = {}

        # Provide the tags of the very own object (ScenarioCategory).
        if self.tags:
            tags["{:s}::ScenarioCategory".format(self.name)] = self.tags

        # Provide the tags for each DynamicPhysicalThingCategory.
        tags = derive_actor_tags(self.dynamic_physical_things, self.acts, tags=tags)

        # Provide the tags for each ActorCategory.
        tags = derive_actor_tags(self.actors, self.acts, tags=tags)

        # Provide the tags of the StaticPhysicalThingCategories.
        for static_physical_thing in self.static_physical_things:
            if static_physical_thing.tags:
                tags["{:s}::StaticPhysicalThingCategory".format(static_physical_thing.name)] = \
                    static_physical_thing.tags

        # Return the tags.
        return tags

    def includes(self, scenario_category: ScenarioCategory) -> bool:
        """ Determine whether the this scenario category "includes" the given scenario category.

        It is checked whether the passed ScenarioCategory is included in this
        scenario category. To determine whether this is the case, the derived
        tags are used. The derived tags from this scenario category should be at
        least present (or subtags of the tags) in the provided ScenarioCategory.

        :param scenario_category: The potential ScenarioCategory that is
            included in this scenario category.
        :return: Whether or not the ScenarioCategory is included.
        """
        # Determine the derived tags of this and the other scenario category.
        own_tags = self.derived_tags()
        other_tags = scenario_category.derived_tags()

        # Check for tags directly related to the ScenarioCategory. These tags should be directly
        # present for the scenario.
        if not _check_tags(own_tags, other_tags, "ScenarioCategory", "ScenarioCategory"):
            return False

        # Check for tags related to the StaticEnvironment.
        if not _check_tags(own_tags, other_tags, "StaticEnvironmentCategory",
                           "StaticEnvironmentCategory"):
            return False

        # Check for the actors, dynamic physical things, and static physical things.
        if not _check_multiple_tags(own_tags, other_tags, "ActorCategory") or \
                not _check_multiple_tags(own_tags, other_tags, "DynamicPhysicalThingCategory") or \
                not _check_multiple_tags(own_tags, other_tags, "StaticPhysicalThingCategory"):
            return False
        return True

    def __str__(self) -> str:
        """ Method that will be called when printing the scenario category.

        :return: string to print.
        """

        # Show the name
        string = "Name: {:s}\n".format(self.name)

        # Show the description of the scenario class
        string += "Description:\n"
        words = self.description.split(' ')
        line = ""
        for word in words:
            if len(line) + len(word) <= self.maxprintlength:
                line += "  {:s}".format(word)
            else:
                string += "{:s}\n".format(line)
                line = "  {:s}".format(word)
        if line:
            string += "{:s}\n".format(line)

        # Show the tags
        string += "Tags:\n"
        if self.tags is None:
            string += "Not available\n"
        else:
            for tag in self.tags:
                string += u"\u2502\u2500 {:s}\n".format(tag)
        derived_tags = self.derived_tags()
        for i, (key, tags) in enumerate(derived_tags.items(), start=1):
            string += u"{}\u2500 {:s}\n".format(u"\u2514" if i == len(derived_tags) else u"\u251C",
                                                key)
            for j, tag in enumerate(tags, start=1):
                string += "{}  {}\u2500 {:s}\n".format(" " if i == len(derived_tags) else u"\u2502",
                                                       "\u2514" if j == len(tags) else "\u251C",
                                                       tag)

        return string

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of
        ScenarioCategory to JSON.

        :return: dictionary that can be converted to a json file.
        """
        scenario_category = QualitativeThing.to_json(self)
        scenario_category["image"] = self.image
        scenario_category["static_physical_thing_category"] = \
            [{'name': thing.name, 'uid': thing.uid} for thing in self.static_physical_things]
        scenario_category["dynamic_physical_thing_category"] = \
            [{'name': thing.name, 'uid': thing.uid} for thing in self.dynamic_physical_things]
        scenario_category["actor_category"] = [{'name': actor.name, 'uid': actor.uid}
                                               for actor in self.actors]
        scenario_category["activity_category"] = [{'name': activity.name, 'uid': activity.uid}
                                                  for activity in self.activities]
        scenario_category["act"] = []
        for dynamic_thing, activity in self.acts:
            key_name = "actor" if isinstance(dynamic_thing, ActorCategory) else "dynamic_thing"
            scenario_category["act"].append({key_name: dynamic_thing.uid, "activity": activity.uid})
        scenario_category["derived_tags"] = self.derived_tags()
        for key, tags in scenario_category["derived_tags"].items():
            scenario_category["derived_tags"][key] = [tag.to_json() for tag in tags]
        return scenario_category

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
        scenario_category = self.to_json()
        scenario_category["static_physical_thing_category"] = \
            [thing.to_json_full() for thing in self.static_physical_things]
        scenario_category["dynamic_physical_thing_category"] = \
            [thing.to_json_full() for thing in self.dynamic_physical_things]
        scenario_category["actor_category"] = [actor.to_json_full() for actor in
                                               self.actors]
        scenario_category["activity_category"] = [activity.to_json_full() for activity in
                                                  self.activities]
        return scenario_category


def scenario_category_from_json(json: dict) -> ScenarioCategory:
    """ Get ScenarioCategory object from JSON code.

    It is assumed that all the attributes are fully defined. Hence, the
    StaticEnvironmentCategory, all ActorCategory, and all ActivityCategory need
    to be defined, instead of only a reference to their IDs.

    :param json: JSON code of ScenarioCategory.
    :return: ScenarioCategory object.
    """
    scenario_category = ScenarioCategory(json["image"], description=json["description"],
                                         name=json["name"], uid=int(json["id"]),
                                         tags=[tag_from_json(tag) for tag in json["tag"]])
    static_physical_things = [static_physical_thing_category_from_json(thing)
                              for thing in json["static_physical_thing_category"]]
    scenario_category.set_static_physical_things(static_physical_things)
    dynamic_physical_things = [dynamic_physical_thing_category_from_json(thing)
                               for thing in json["dynamic_physical_thing_category"]]
    scenario_category.set_dynamic_physical_things(dynamic_physical_things)
    actors = [actor_category_from_json(actor) for actor in json["actor_category"]]
    scenario_category.set_actors(actors)
    activities = [activity_category_from_json(activity)
                  for activity in json["activity_category"]]
    scenario_category.set_activities(activities)
    dynamic_thing_uids = [dynamic_thing.uid for dynamic_thing in dynamic_physical_things]
    actor_uids = [actor.uid for actor in actors]
    activity_uids = [activity.uid for activity in activities]
    acts = []
    for act in json["act"]:
        if "actor" in act:
            dynamic_thing = actors[actor_uids.index(act["actor"])]
        else:
            dynamic_thing = dynamic_physical_things[dynamic_thing_uids.index(act["dynamic_thing"])]
        acts.append((dynamic_thing, activities[activity_uids.index(act["activity"])]))
    scenario_category.set_acts(acts)
    return scenario_category


def derive_actor_tags(dynamic_things: List, acts: List, tags: dict = None) -> dict:
    """ Derive the tags that are associated with the actors.

    The tags of an DynamicPhysicalThing(Category) or Actor(Category) will be
    added to the dictionary "tags". The key equals <name of actor>::<class>,
    where class is supposed to be either DynamicPhysicalThing,
    DynamicPhysicalThingCategory, Actor, or ActorCategory, whereas the value
    will be the list of tags that are associated with the dynamic physical
    thing.

    :param dynamic_things: The dynamic physical things of the
        Scenario(Category).
    :param acts: The acts of the Scenario(Category).
    :param tags: Initial tags that will be amended with tags of the actors.
    :return: Dictionary with each actor as a key and the corresponding values
        denote the tags.
    """
    # By default, tags is an empty dictionary
    if tags is None:
        tags = {}

    for dynamic_thing in dynamic_things:
        actor_tags = dynamic_thing.get_tags()
        for act in acts:
            if act[0] == dynamic_thing:
                actor_tags += act[1].get_tags()
        if actor_tags:
            if isinstance(dynamic_thing, ActorCategory):
                class_name = "ActorCategory"
            elif isinstance(dynamic_thing, Actor):
                class_name = "Actor"
            elif isinstance(dynamic_thing, DynamicPhysicalThingCategory):
                class_name = "DynamicPhysicalThingCategory"
            elif isinstance(dynamic_thing, DynamicPhysicalThing):
                class_name = "DynamicPhysicalThing"
            else:
                raise TypeError("Dynamic physical thing is of type " +
                                "'{}' while it should be of type ".format(type(dynamic_thing)) +
                                "DynamicPhysicalThingCategory, DynamicPhysicalThing, " +
                                "ActorCategory, or Actor.")
            key = "{:s}::{:s}".format(dynamic_thing.name, class_name)
            i = 1
            while key in tags:  # Make sure that a unique key is used.
                i += 1
                key = "{:s}{:d}::{:s}".format(dynamic_thing.name, i, class_name)
            tags[key] = list(set(actor_tags))  # list(set()) makes sure that tags are unique.
    return tags


def create_unique_ids(items: List) -> None:
    """ Update the uids of the items.

    It is assumed that each item has the attribute "uid". If a "uid" is negative
    or similar to another "uid", the "uid" is given the value of (N+1), where
    N is the (up-to-then) highest "uid" or 0 incase the highest "uid" is
    negative.

    :param items: List of the items.
    """
    highest_uid = np.max([item.uid for item in items]).astype(np.int)
    if highest_uid < 0:
        highest_uid = 0
    uids = []
    for item in items:
        if item.uid < 0 or item.uid in uids:
            highest_uid += 1
            item.uid = highest_uid
        uids.append(item.uid)


def _check_tags(tags: dict, subtags: dict, tags_class: str = "ScenarioCategory",
                subtags_class: str = "ScenarioCategory") -> bool:
    """ Check whether (sub)tags of <tags> are present in <subtags>.

    The tags are provided as dictionaries, where each item corresponds to the
    tags of one object that is part of the scenario (category). This function
    checks whether all tags in <tags> of the class <tags_class> are present in
    the tags in <subtags> of the class <subtags_class> (or subtags of the
    <tags>).

    :param tags: Dictionary of the derived tags of the ScenarioCategory.
    :param subtags: Dictionary of the derived tags of the Scenario.
    :param tags_class: Specify attribute to be used of the ScenarioCategory.
    :param subtags_class: Specify attribute to be used of the Scenario.
    :return: Whether the tags of the ScenarioCategory are found in the tags
        of the Scenario.
    """
    sc_keys = fnmatch.filter(tags, "*::{:s}".format(tags_class))
    if sc_keys:  # In this case, there are tags in <tags> related to <tags_class>.
        s_keys = fnmatch.filter(subtags, "*::{:s}".format(subtags_class))
        if s_keys:  # There are tags in <subtags> related to <subtags_class>.
            for tag in tags[sc_keys[0]]:
                if not any(map(tag.is_supertag_of, subtags[s_keys[0]])):
                    return False  # A tag of <tags> is not found in the <subtags>.
        else:  # There are no tags in <subtags> related to <subtags_class>.
            return False
    return True


def _check_multiple_tags(own_tags: dict, other_tags: dict, attribute: str) -> bool:
    """ Check if all tags in <own_tags> are present in <other_tags>.

    This is done for a specific attribute (e.g., actor_categories). For example,
    with actor categories, there is a list made, where each item is a list
    itself with the tags of the corresponding actor category. For each the
    actor category in <own_tags>, there needs to be a (different) actor category
    in <other_tags> that has the same tags (or more tags, or corresponding
    subtags).

    :param own_tags: The tags of the own scenario category.
    :param other_tags: The tags of the scenario category that is potentially
        'included' in the former.
    :param attribute: Check for which attribute we need to check.
    """
    own_actors = fnmatch.filter(own_tags, "*::{:s}".format(attribute))
    other_actors = fnmatch.filter(other_tags, "*::{:s}".format(attribute))
    if len(own_actors) > len(other_actors):  # There must be equal or more actors in other SC.
        return False

    # Create a boolean matrix, where the (i,j)-th element is True if the i-th actor of the
    # provided ScenarioCategory might correspond to the j-th actor of our own scenario category.
    match = np.zeros((len(other_actors), len(own_actors)), dtype=np.bool)
    for i, other_actor in enumerate(other_actors):
        for j, own_actor in enumerate(own_actors):
            match[i, j] = all(any(map(tag.is_supertag_of, other_tags[other_actor]))
                              for tag in own_tags[own_actor])

    # Check if all actors of this scenario category can be matched with the actos in the
    # provided ScenarioCategory
    return _check_match_matrix(match)


def _check_match_matrix(match: np.array) -> bool:
    # The matching of the actors need to be done. If a match is found, the corresponding
    # row and column will be removed from the match matrix.
    n_matches = 1  # Number of matches to look for.
    while match.size:
        # If there is at least one ActorCategory left that has no match, a False will be
        # returned.
        if not all(np.any(match, axis=1)):
            return False

        sum_match_actor = np.sum(match, axis=0)
        j = next((j for j in range(match.shape[1]) if sum_match_actor[j] == n_matches), -1)
        if j >= 0:  # We found an actor of our own scenario category with n matches.
            i = next(i for i in range(match.shape[0]) if match[i, j])
        else:
            sum_match_actor = np.sum(match, axis=1)
            i = next((i for i in range(match.shape[0]) if sum_match_actor[i] == n_matches), -1)
            if i >= 0:  # We found an actor of the ScenarioCategory with n matches
                j = next(j for j in range(match.shape[1]) if match[i, j])
            else:
                # Try again for higher n (number of matches)
                n_matches = n_matches + 1
                continue
        match = np.delete(np.delete(match, i, axis=0), j, axis=1)
        n_matches = 1
    return True
