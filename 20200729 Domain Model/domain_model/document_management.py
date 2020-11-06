""" Class DocumentManagement

Creation date: 2020 03 23
Author(s): Erwin de Gelder

Modifications:
2020 11 01: Update class based on the update of the domain model.
"""

from typing import Callable, List, NamedTuple, Union
import json
from .actor import Actor, actor_from_json
from .actor_category import ActorCategory, actor_category_from_json
from .activity import Activity, activity_from_json
from .activity_category import ActivityCategory, activity_category_from_json
from .event import Event, event_from_json
from .model import Model, model_from_json
from .physical_element import PhysicalElement, physical_element_from_json
from .physical_element_category import PhysicalElementCategory, physical_element_category_from_json
from .scenario import Scenario, scenario_from_json
from .scenario_category import ScenarioCategory, scenario_category_from_json
from .scenario_element import get_empty_dm_object, DMObjects


PossibleObject = NamedTuple("PossibleObject", [("type", object), ("from_json", Callable)])


class DocumentManagement:
    """ DocumentManagement

    This class is used to read and write objects from the scenario database.

    A path can be provided during instantiating DocumentManagement. If this path
    contains the output of the `to_json` function, it will read its contents.

    Attributes:
        version (str): Version number of the DocumentManagement.
        possible_objects (dict): Dictionary containing the class names of the
            objects for which the JSON code can be retrieved from the database.
        collections (dict): All JSON codes are contained here.
        realizations (DMObjects): All objects that are instantiated are
            contained here.
    """
    def __init__(self, path_or_realizations: [str, DMObjects] = None):
        # Version of the DocumentManagement. This is used as meta information for the documents.
        self.version = "0.2"

        # Define the list of objects that we can work with.
        self.possible_objects = dict(
            actor=PossibleObject(type=Actor, from_json=self._actor_from_json),
            actor_category=PossibleObject(type=ActorCategory, from_json=actor_category_from_json),
            activity=PossibleObject(type=Activity, from_json=self._activity_from_json),
            activity_category=PossibleObject(type=ActivityCategory,
                                             from_json=self._activity_category_from_json),
            event=PossibleObject(type=Event, from_json=event_from_json),
            model=PossibleObject(type=Model, from_json=model_from_json),
            physical_element=PossibleObject(type=PhysicalElement,
                                            from_json=self._physical_element_from_json),
            physical_element_category=PossibleObject(type=PhysicalElementCategory,
                                                     from_json=physical_element_category_from_json),
            scenario=PossibleObject(type=Scenario, from_json=self._scenario_from_json),
            scenario_category=PossibleObject(type=ScenarioCategory,
                                             from_json=self._scenario_category_from_json))

        # Create an empty "database"
        self.collections = dict()
        self.realizations = get_empty_dm_object()
        for possible_object in self.possible_objects:
            self.collections[possible_object] = dict()

        if path_or_realizations is not None:
            if isinstance(path_or_realizations, str):
                # Load the collections.
                self.from_json(path_or_realizations)
            elif isinstance(path_or_realizations, DMObjects):
                # Convert all items in the input to JSON code.
                for key in self.possible_objects:
                    for item in getattr(path_or_realizations, key).values():
                        self.add_item(item)

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

    def add_item(self, item: Union[Actor, ActorCategory, Activity, ActivityCategory, Event, Model,
                                   PhysicalElement, PhysicalElementCategory, Scenario,
                                   ScenarioCategory], overwrite: bool = True) -> None:
        """ Write an item to the database.

        If an similar object with the same ID is already stored in the database,
        a message will be returned in case overwrite=False. In this case, the
        object is not written to the database. When overwrite=True and a similar
        object with the ID is already stored in the database, the document is
        updated.

        :param item: The item that has to be written to the database.
        :param overwrite: Whether to overwrite the database entry.
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
        if not overwrite and item.uid in self.collections[collection]:
            print("Warning: object with same ID is already in database.")
            return

        # Write object to the database.
        json_code = item.to_json()
        json_code['_version'] = self.version
        self.collections[collection][item.uid] = json_code

    def delete_item(self, name: str, uid: int) -> None:
        """ Delete an item of the database.

        :param name: Name of the object.
        :param uid: The ID.
        """
        del self.collections[name][uid]

    def get_item(self, name: str, uid: int):
        """ Obtain an item of the database.

        :param name: Name of the object.
        :param uid: The ID.
        :return: The item.
        """
        if uid in getattr(self.realizations, name):
            return getattr(self.realizations, name)[uid]

        return self.possible_objects[name].from_json(self.collections[name][uid],
                                                     self.realizations)

    def get_ordered_item(self, name: str, i: int):
        """ Obtain the i-th item, based on sorted IDs.

        :param name: Name of the object.
        :param i: The i-th item.
        :return: The item.
        """
        if i >= len(self.collections[name]):
            raise ValueError("Requesting item i={:d}, but only {:d} items available"
                             .format(i, len(self.collections[name])))
        uid = sorted(self.collections[name].keys())[i]
        return self.get_item(name, uid)

    def _actor_from_json(self, json_code: dict, realizations: DMObjects):
        actor_category = self.get_item("actor_category", json_code["category"]["uid"])
        return actor_from_json(json_code, realizations, category=actor_category)

    def _activity_from_json(self, json_code: dict, realizations: DMObjects):
        activity_category = self.get_item("activity_category", json_code["category"]["uid"])
        start = self.get_item("event", json_code["start"]["uid"])
        end = self.get_item("event", json_code["end"]["uid"])
        return activity_from_json(json_code, realizations, start=start, end=end,
                                  category=activity_category)

    def _activity_category_from_json(self, json_code: dict, realizations: DMObjects):
        model = self.get_item("model", json_code["model"]["uid"])
        return activity_category_from_json(json_code, realizations, model=model)

    def _physical_element_from_json(self, json_code: dict, realizations: DMObjects):
        physical_element_category = self.get_item("physical_element_category",
                                                  json_code["category"]["uid"])
        return physical_element_from_json(json_code, realizations, category=physical_element_category)

    def _scenario_from_json(self, json_code: dict, realizations: DMObjects):
        actors = [self.get_item("actor", actor["uid"]) for actor in json_code["actor"]]
        activities = [self.get_item("activity", activity["uid"])
                      for activity in json_code["activity"]]
        physical_elements = [self.get_item("physical_element", physical_element["uid"])
                             for physical_element in json_code["physical_elements"]]
        start = self.get_item("event", json_code["start"]["uid"])
        end = self.get_item("event", json_code["end"]["uid"])
        return scenario_from_json(json_code, realizations, actors=actors, activities=activities,
                                  physical_elements=physical_elements, start=start, end=end)

    def _scenario_category_from_json(self, json_code: dict, realizations: DMObjects):
        actor_categories = [self.get_item("actor_category", actor_category["uid"])
                            for actor_category in json_code["actor_categories"]]
        activity_categories = [self.get_item("activity_category", activity_category["uid"])
                               for activity_category in json_code["activity_categories"]]
        physical_element_categories = [self.get_item("physical_element_category",
                                                     physical_element_category["uid"])
                                       for physical_element_category in
                                       json_code["physical_element_categories"]]
        return scenario_category_from_json(json_code, realizations, actors=actor_categories,
                                           activities=activity_categories,
                                           physical_elements=physical_element_categories)
