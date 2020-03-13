"""
Class DocumentManagement


Author
------
Erwin de Gelder

Creation
--------
14 Nov 2018

To do
-----

Modifications
-------------
22 Nov 2018 Enable instantiation of various objects. Add query on name.
30 Nov 2018 Make it possible to update documents. Meta information is added to documents.
04 Dec 2018 Change several methods to private methods.
05 Dec 2018 Change database id to tno_test.

"""

from typing import List
import pydocumentdb.document_client as document_client
from .actor_category import ActorCategory, actor_category_from_json
from .activity_category import ActivityCategory, activity_category_from_json
from .static_environment_category import StaticEnvironmentCategory, stat_env_category_from_json
from .scenario_category import ScenarioCategory, scenario_category_from_json
from .static_environment import StaticEnvironment, stat_env_from_json
from .actor import Actor, actor_from_json
from .activity import Activity, activity_from_json
from .scenario import Scenario, scenario_from_json


class DocumentManagement:
    """ DocumentManagement

    This class is used to read and write objects from the scenario database.
    # TODO: Add author.

    Attributes:
        version (str): Version number of the DocumentManagement.
        client (DcoumentClient): Client that is used to manage the database.
        ids (dict): Dictionary containing all IDs that are used.
        possible_objects (dict): Dictionary containing the class names of the objects for which
            the JSON code can be retrieved from the database.
    """
    def __init__(self, uri: str, primary_key: str):
        # Version of the DocumentManagement. This is used as meta information for the documents.
        self.version = "0.1"

        # Create the client that is used to manage the database.
        self.client = document_client.DocumentClient(uri, {'masterKey': primary_key})

        # Define all different ids.
        self.ids = {"database_id": "tno_test",
                    "collection_ids": "last_ids",
                    "ActorCategory": "actor_categories",
                    "ActivityCategory": "activity_categories",
                    "StaticEnvironmentCategory": "static_environment_categories",
                    "ScenarioCategory": "scenario_categories",
                    "Actor": "actors",
                    "Activity": "activities",
                    "StaticEnvironment": "static_environments",
                    "Scenario": "scenarios"}

        # Define the list of objects that we can work with.
        self.possible_objects = \
            {"ActorCategory": {"type": ActorCategory,
                               "from_json": actor_category_from_json},
             "ActivityCategory": {"type": ActivityCategory,
                                  "from_json": activity_category_from_json},
             "StaticEnvironmentCategory": {"type": StaticEnvironmentCategory,
                                           "from_json": stat_env_category_from_json},
             "ScenarioCategory": {"type": ScenarioCategory,
                                  "from_json": self._get_scenario_class_from_json},
             "Actor": {"type": Actor,
                       "from_json": self._get_actor_from_json},
             "Activity": {"type": Activity,
                          "from_json": self._get_activity_from_json},
             "StaticEnvironment": {"type": StaticEnvironment,
                                   "from_json": self._get_stat_env_from_json},
             "Scenario": {"type": Scenario,
                          "from_json": self._get_scenario_from_json}}

    def get_last_id(self, collection_id: str) -> int:
        """ Get the last ID that is used in the database.

        The last ID that is used for a specific collection is retrieved. If the collection_id
        does not exist, an error with status code 404 is thrown.

        :param collection_id: ID of the collection.
        :return: ID of last added document.
        """
        # Try to read the appropriate document.
        collection_link = "dbs/" + self.ids["database_id"] + "/colls/" + self.ids["collection_ids"]
        response = self.client.ReadDocument(collection_link + "/docs/" + collection_id)
        return response["value"]

    def update_last_id(self, collection_id: str) -> int:
        """ Update and return the last ID that is used in the database.

        The last ID that is used for a specific collection is retrieved. The update is simply
        done by adding 1 (one). If the collection_id does not exist, an error with status code
        404 is thrown.

        :param collection_id: ID of the collection.
        :return: the updated ID is returned.
        """
        # Retrieve the last_id.
        last_id = self.get_last_id(collection_id)

        # Update the last_id.
        last_id += 1

        # Update the database.
        collection_link = "dbs/" + self.ids["database_id"] + "/colls/" + self.ids["collection_ids"]
        document_link = collection_link + "/docs/" + collection_id
        document = {"id": collection_id, "value": last_id}
        self.client.ReplaceDocument(document_link, document)

        # Return the last_id.
        return last_id

    def write_dm_object(self, dm_object, overwrite=True) -> None:
        """ Write an object from the domain model to the database

        The domain model object can be one of the following objects:
        - ActorCategory
        - ActivityCategory
        - StaticEnvironmentCategory
        - ScenarioCategory
        - Actor
        - Activity
        - StaticEnvironment
        - Scenario

        If the unique id is set to -1, the ID is automatically generated. If an
        similar object with the same ID is already stored in the database, a message
        will be returned in case overwrite=False. In this case, the object is not written to the
        database. When overwrite=True and a similar object with the ID is already stored in the
        database, the document is updated.

        :param dm_object: the object that is written to the database.
        :param overwrite: Whether to overwrite the database entry.
        """
        # Check whether input is correctly defined.
        is_correct_object = False
        str_object = ""
        for str_object, possible_object in self.possible_objects.items():
            if isinstance(dm_object, possible_object["type"]):
                is_correct_object = True
                break
        if not is_correct_object:
            raise TypeError("Domain model object is of type '{0}' and this type is not supported.".
                            format(type(dm_object)))

        # Check if we need to generate an ID.
        if dm_object.uid == -1:
            dm_object.uid = self.update_last_id(self.ids[str_object])
        else:
            last_id = self.get_last_id(self.ids[str_object])
            if dm_object.uid > last_id + 1:
                # In this case, the document ID is larger than it would be when the default is
                # used, i.e., if uid == -1. This is not desired, as it will mess up the
                # database. Hence, the uid is chosen as if uid == -1.
                updated_id = self.update_last_id(self.ids[str_object])
                print("Unique ID automatically changed from {:d} to {:d}.".format(dm_object.uid,
                                                                                  updated_id))
                dm_object.uid = updated_id
            elif dm_object == last_id + 1:
                self.update_last_id(self.ids[str_object])

        # Check if ID already exist.
        collection_link = "dbs/" + self.ids["database_id"] + "/colls/" + self.ids[str_object]
        query = {"query": "SELECT a.id FROM {:s} a WHERE a.id='{:d}'"
                          .format(self.ids[str_object], dm_object.uid)}
        results = list(self.client.QueryDocuments(collection_link, query))
        if results:
            if overwrite:
                document_link = "{:s}/docs/{:d}".format(collection_link, dm_object.uid)
                self.client.ReplaceDocument(document_link, self.dm_object_document(dm_object))
            else:
                print("{:s} with uid={:d} already exists.".format(str_object, dm_object.uid))
        else:
            # Write the object to the database.
            self.client.CreateDocument(collection_link, self.dm_object_document(dm_object))

    def dm_object_document(self, dm_object) -> dict:
        """ Convert domain model object to dictionary (= Cosmos document)

        Use is made of the object's to_json() method. Furthermore, some additional (meta)
        information is added.
        The following information is added:
         - Version of the DocumentManagement;
         - Date of writing document to database.
         - Time of writing document to database.

        :param dm_object: the object that is written to the database.
        :return: JSON code that can be written to database.
        """
        document = dm_object.to_json()
        document["_version"] = self.version
        # TODO: write date and time.
        return document

    def get_json_by_id(self, str_object: str, uid: int) -> dict:
        """ Get JSON code from ID

        :param str_object: name of the class of the object.
        :param uid: The unique ID of the object.
        :return: The object that is queried.
        """
        # Check if the required object is feasible, i.e., if it is listed in the possible_objects.
        if str_object not in self.possible_objects:
            raise ValueError("Listed object '{:s}' not (yet) supported.".format(str_object))

        # Query the object.
        collection_link = "dbs/" + self.ids["database_id"] + "/colls/" + self.ids[str_object]
        query = {"query": "SELECT * FROM {:s} a WHERE a.id='{:d}'"
                          .format(self.ids[str_object], uid)}
        results = list(self.client.QueryDocuments(collection_link, query))

        # Check if there is an object with the given id.
        if not results:
            raise ValueError("No object of type '{:s}' with id '{:d}'".format(str_object, uid))

        # If we have results, return the object.
        return results[0]

    def get_object_by_id(self, str_object: str, uid: int):
        """ Get object from its id

        :param str_object: name of the class of the object.
        :param uid: The unique ID of the object.
        :return: The object that is queried.
        """
        # Get the JSON code from the database.
        json = self.get_json_by_id(str_object, uid)

        # Return the object.
        return self.possible_objects[str_object]["from_json"](json)

    def get_jsons_by_name(self, str_object: str, name: str) -> List[dict]:
        """ Get JSON code(s) from name

        It is possible that there are no objects with the given name. In that case, an empty
        list is returnd. It is also possible that there are multiple objects with the given
        name. In any case, a list is returned.

        :param str_object: name of the class of the object.
        :param name: The name of the object(s).
        :return: A list of JSON codes.
        """
        # Check if the required object is feasible, i.e., if it is listed in the possible_objects.
        if str_object not in self.possible_objects:
            raise ValueError("Listed object '{:s}' not (yet) supported.".format(str_object))

        # Query the object.
        collection_link = "dbs/" + self.ids["database_id"] + "/colls/" + self.ids[str_object]
        query = {"query": "SELECT * FROM {:s} a WHERE a.name='{:s}'"
                          .format(self.ids[str_object], name)}
        return list(self.client.QueryDocuments(collection_link, query))

    def get_objects_by_name(self, str_object: str, name: str) -> list:
        """ Get object(s) from name

        It is possible that there are no objects with the given name. In that case, an empty
        list is returnd. It is also possible that there are multiple objects with the given
        name. In any case, a list is returned.

        :param str_object: name of the class of the object.
        :param name: The name of the object(s).
        :return: List of the object(s) that are queried.
        """
        # Get the JSON code from the database.
        jsons = self.get_jsons_by_name(str_object, name)

        # Return the object(s).
        return [self.possible_objects[str_object]["from_json"](json) for json in jsons]

    def _get_scenario_class_from_json(self, json: dict) -> ScenarioCategory:
        """ Create ScenarioCategory from the JSON code

        A ScenarioCategory cannot be directly instantiated from the JSON code obtained from the
        database, because it contains attributes that are stored in the database. Hence,
        these attributes need to be retrieved from the database first.

        :param json: The JSON code that is retrieved from the database.
        :return: The object that is queried
        """
        # Get the JSON code of the actors, activities, and static environment.
        json["actor_category"] = [self.get_json_by_id("ActorCategory", uid=actor["uid"])
                                  for actor in json["actor_category"]]
        json["activity_category"] = [self.get_json_by_id("ActivityCategory", uid=actor["uid"])
                                     for actor in json["activity_category"]]
        json["static_environment_category"] = \
            self.get_json_by_id("StaticEnvironmentCategory",
                                json["static_environment_category"]["uid"])
        return scenario_category_from_json(json)

    def _get_stat_env_from_json(self, json: dict) -> StaticEnvironment:
        """ Create StaticEnvironment from the JSON code

        A StaticEnvironment cannot be directly instantiated from the JSON code obtained from the
        database, because it contains attributes that are stored in the database. Hence,
        these attributes need to be retrieved from the database first.

        :param json: The JSON code that is retrieved from the database.
        :return: The object that is queried
        """
        # Get JSON code of the StaticEnvironmentCategory.
        json["static_environment_category"] = \
            self.get_json_by_id("StaticEnvironmentCategory",
                                json["static_environment_category"]["uid"])
        return stat_env_from_json(json)

    def _get_actor_from_json(self, json: dict) -> Actor:
        """ Create Actor from the JSON code

        An Actor cannot be directly instantiated from the JSON code obtained from the
        database, because it contains attributes that are stored in the database. Hence,
        these attributes need to be retrieved from the database first.

        :param json: The JSON code that is retrieved from the database.
        :return: The object that is queried
        """
        # Get JSON code of the ActorCategory.
        json["actor_category"] = self.get_json_by_id("ActorCategory",
                                                     json["actor_category"]["uid"])
        return actor_from_json(json)

    def _get_activity_from_json(self, json: dict) -> Activity:
        """ Create Activity from the JSON code

        An Activity cannot be directly instantiated from the JSON code obtained from the
        database, because it contains attributes that are stored in the database. Hence,
        these attributes need to be retrieved from the database first.

        :param json: The JSON code that is retrieved from the database.
        :return: The object that is queried
        """
        # Get JSON code of the ActivityCategory.
        json["activity_category"] = self.get_json_by_id("ActivityCategory",
                                                        json["activity_category"]["uid"])
        return activity_from_json(json)

    def _get_scenario_from_json(self, json: dict) -> Scenario:
        """ Create Scenario from the JSON code

        A Scenario cannot be directly instantiated from the JSON code obtained from the
        database, because it contains attributes that are stored in the database. Hence,
        these attributes need to be retrieved from the database first.

        :param json: The JSON code that is retrieved from the database.
        :return: The object that is queried
        """
        # Get JSON code of the actors and their corresponding categories.
        # If the ActorCategory is already loaded before (based on the unique ID), it will be
        # skipped.
        json["actor"] = [self.get_json_by_id("Actor", actor["uid"]) for actor in json["actor"]]
        actor_uids = []
        for actor in json["actor"]:
            actor_uid = actor["actor_category"]["uid"]
            if actor_uid not in actor_uids:
                actor["actor_category"] = self.get_json_by_id("ActorCategory", actor_uid)
                actor_uids.append(actor_uid)

        # Get JSON code of the activities and their corresponding categories.
        # If the ActivityCategory is already loaded before (based on the unique ID), it will be
        # skipped.
        json["activity"] = [self.get_json_by_id("Activity", activity["uid"])
                            for activity in json["activity"]]
        activity_uids = []
        for activity in json["activity"]:
            activity_uid = activity["activity_category"]["uid"]
            if activity_uid not in activity_uids:
                activity["activity_category"] = self.get_json_by_id("ActivityCategory",
                                                                    activity_uid)
                activity_uids.append(activity_uid)

        # Get the JSON code of the StaticEnvironment and its StaticEnvironmentCategory.
        json["static_environment"] = self.get_json_by_id("StaticEnvironment",
                                                         json["static_environment"]["uid"])
        json["static_environment"]["static_environment_category"] = \
            self.get_json_by_id("StaticEnvironmentCategory",
                                json["static_environment"]["static_environment_category"]["uid"])
        return scenario_from_json(json)
