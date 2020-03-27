""" Object that can contain data as if it is a scenario database

Creation date: 2020 03 23
Author(s): Erwin de Gelder

Modifications:
"""

import os
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
            actor=PossibleObject(type=Actor, from_json=self._actor_from_json),
            activity=PossibleObject(type=Activity, from_json=self._activity_from_json),
            static_environment=PossibleObject(type=StaticEnvironment,
                                              from_json=self._stat_env_from_json),
            scenario=PossibleObject(type=Scenario, from_json=self._scenario_from_json))

        # Create an empty "database".
        self.collections = dict()
        self.realizations = dict()
        for possible_object in self.possible_objects:
            self.collections[possible_object] = []
            self.realizations[possible_object] = dict()

        if path is not None:
            # Load the collections.
            self.from_json(path)

    def to_json(self, path: str, **kwargs) -> None:
        """ Store the 'database' in a json file.

        :param path: The filename to which to store the file to.
        :param kwargs: Additional parameters that will be parsed to json.dump().
        """
        with open(path, "w") as file:
            json.dump(self.collections, file, **kwargs)

    def from_json(self, path: str) -> None:
        """ Read a 'database' from a json file.

        :param path: The filename of the database.
        """
        with open(path, "r") as file:
            self.collections = json.load(file)

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

    def get_json(self, name: str, uid: int) -> object:
        """ Obtain an item of the database.

        :param name: Name of the object.
        :param uid: The ID.
        :return: The json code of the item.
        """
        if name not in self.possible_objects:
            raise KeyError("Invalid 'name'.")
        if uid > len(self.collections[name]):
            raise KeyError("ID does not exist.")
        return self.collections[name][uid]

    def get_item(self, name: str, uid: int):
        """ Obtain an item of the database.

        :param name: Name of the object.
        :param uid: The ID.
        :return: The item.
        """
        if uid in self.realizations[name]:
            return self.realizations[name][uid]
        item = self.possible_objects[name].from_json(self.get_json(name, uid))
        self.realizations[name][uid] = item
        return item

    def _actor_from_json(self, json_code: dict):
        # Obtain the ActorCategory.
        actor_category = self.get_item("actor_category", json_code["actor_category"]["uid"])
        return actor_from_json(json_code, actor_category=actor_category)

    def _activity_from_json(self, json_code: dict):
        # Obtain the ActivityCategory.
        activity_category = self.get_item("activity_category",
                                          json_code["activity_category"]["uid"])
        return activity_from_json(json_code, activity_category=activity_category)

    def _stat_env_from_json(self, json_code: dict):
        # Obtain the StaticScenarioCategory.
        stat_env_category = self.get_item("static_environment_category",
                                          json_code["static_environment_category"]["uid"])
        return stat_env_from_json(json_code, stat_env_category=stat_env_category)

    def _scenario_from_json(self, json_code: dict):
        # Obtain the static environment, actors, and activities.
        static_environment = self.get_item("static_environment",
                                           json_code["static_environment"]["uid"])
        actors = [self.get_item("actor", actor["uid"]) for actor in json_code["actor"]]
        activities = [self.get_item("activity", activity["uid"])
                      for activity in json_code["activity"]]
        return scenario_from_json(json_code, static_environment=static_environment,
                                  actors=actors, activities=activities)


if __name__ == "__main__":
    DBE = DataBaseEmulator(os.path.join("data", "5_cutin_scenarios", "database_small.json"))
    x = DBE.get_item("scenario", 0)
    print(json.dumps(x.to_json_full(), indent=4))
