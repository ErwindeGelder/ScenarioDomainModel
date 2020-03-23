""" Object that can contain data as if it is a scenario database

Creation date: 2020 03 23
Author(s): Erwin de Gelder

Modifications:
"""

from typing import Callable, NamedTuple, Union
import json
from domain_model import ActorCategory, ActivityCategory, StaticEnvironmentCategory, \
    ScenarioCategory, Actor, Activity, StaticEnvironment, Scenario, actor_category_from_json, \
    activity_category_from_json, stat_env_category_from_json, scenario_category_from_json, \
    actor_from_json, activity_from_json, stat_env_from_json, scenario_from_json


PossibleObject = NamedTuple("PossibleObject", [("type", object), ("from_json", Callable)])


class DataBaseEmulator:
    """ Container for a scenario database """
    def __init__(self, path: str = None):
        # Define the different 'collections'.
        self.possible_objects = dict(
            actor_category=PossibleObject(type=ActorCategory, from_json=actor_category_from_json),
            activity_category=PossibleObject(type=ActivityCategory,
                                             from_json=activity_category_from_json),
            static_environment_category=PossibleObject(type=StaticEnvironmentCategory,
                                                       from_json=stat_env_category_from_json),
            scenario_category=PossibleObject(type=ScenarioCategory,
                                             from_json=scenario_category_from_json),
            actor=PossibleObject(type=Actor, from_json=actor_from_json),
            activity=PossibleObject(type=Activity, from_json=activity_from_json),
            static_environment=PossibleObject(type=StaticEnvironment,
                                              from_json=stat_env_from_json),
            scenario=PossibleObject(type=Scenario, from_json=scenario_from_json))
        if path is not None:
            # Load the "database".
            with open(path, "r") as file:
                self.collections = json.load(file)
        else:
            # Create an empty "database".
            self.collections = dict()
            for possible_object in self.possible_objects:
                self.collections[possible_object] = []

    def to_json(self, path: str, **kwargs) -> None:
        """ Store the 'database' in a json file.

        :param path: The filename to which to store the file to.
        :param kwargs: Additional parameters that will be parsed to json.dump().
        """
        with open(path, "w") as file:
            json.dump(self.collections, file, **kwargs)

    def add_item(self, item: Union[ActorCategory, ActivityCategory, StaticEnvironmentCategory,
                                   ScenarioCategory, Actor, Activity, StaticEnvironment,
                                   Scenario]) -> None:
        """ Write an item to the database.

        :param item: The item that has to be written to the database.
        """
        # Obtain name of collection.
        collection = None
        for key, value in self.possible_objects.items():
            if isinstance(item, value.type):
                collection = key
                break
        if collection is None:
            raise TypeError("The provided item is not a valid object to be written to the database")

        # See if we already have an object with this ID.
        if item.uid >= 0:
            print("Warning: object already has an ID, but ID will be overwritten.")

        item.uid = len(self.collections[collection])
        self.collections[collection].append(item.to_json())

    def get_item(self, name: str, uid: int) -> object:
        """ Obtain an item of the database.

        :param name: Name of the object.
        :param uid: The ID.
        :return: The item.
        """
        if name not in self.possible_objects:
            raise KeyError("Invalid 'name'.")
        if uid > len(self.collections[name]):
            raise KeyError("ID does not exist.")
        return self.possible_objects[name].from_json(self.collections[name][uid])
