""" Class ScenarioElement

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 14: When converting to JSON, ID should be a string (otherwise Cosmos DB gives error).
2018 12 06: Make it possible to return full JSON code (incl. attributes' JSON code).
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 13: Update of terminology.
2020 07 31: Return a copy of the list of tags when using get_tags().
2020 08 15: Change Default to ScenarioElement.
2020 08 22: Add function to obtain properties from a dictionary.
2020 08 23: If no uid is given, generate one.
2020 10 04: Provide functions for creating objects from JSON code.
2020 10 12: Remove Static/DynamicPhysicalThing(Category), add PhysicalElement(Category) to objects.
2020 10 25: Change name of Thing to ScenarioElement.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, NamedTuple, Tuple, Union
import uuid
from .tags import Tag, tag_from_json
from .type_checking import check_for_type, check_for_list


class ScenarioElement(ABC):
    """ ScenarioElement that is used for most classes.

    Because most classes contain the attributes 'name' and 'tags', a default
    class is created that contains these attributes. This class also does type
    checking for these attributes. This class supports conversion to JSON. This
    is an abstract class, so it is not possible to instantiate objects from this
    class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    @abstractmethod
    def __init__(self, name: str = "", uid: int = None, tags: List[Tag] = None, **kwargs):
        # Check the types of the inputs
        if uid is not None:
            check_for_type("uid", uid, int)
        check_for_type("name", name, str)
        check_for_list("tags", tags, Tag)

        if uid is None:
            self.uid = uuid.uuid4().int  # type: int
        else:
            self.uid = uid  # type: int
        self.name = name  # type: str
        self.tags = [] if tags is None else tags

    def get_tags(self) -> List[Tag]:
        """ Return the list of tags related to this Object.

        It simply returns the tags associated to this Object.

        :return: List of tags.
        """
        return self.tags.copy()

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the default attributes to JSON.

        :return: dictionary that can be converted to a json file.
        """
        default_class = {"name": self.name,
                         "id": "{:d}".format(self.uid),
                         "tags": [tag.to_json() for tag in self.tags]}
        return default_class

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
        return self.to_json()


def _scenario_element_props_from_json(json: dict) -> dict:
    return dict(name=json["name"],
                uid=int(json["id"]),
                tags=[tag_from_json(tag) for tag in json["tags"]])


DMObjects = NamedTuple("dm_objects", [("event", Dict),
                                      ("physical_element", Dict),
                                      ("physical_element_category", Dict),
                                      ("actor", Dict),
                                      ("actor_category", Dict),
                                      ("activity", Dict),
                                      ("activity_category", Dict),
                                      ("model", Dict),
                                      ("scenario_category", Dict),
                                      ("scenario", Dict)])


def get_empty_dm_object() -> NamedTuple:
    """ Return an empty NamedTuple to store all objects.

    :return: The empty NamedTuple in which all objects can be stored.
    """
    return DMObjects(event=dict(),
                     physical_element=dict(),
                     physical_element_category=dict(),
                     actor=dict(),
                     actor_category=dict(),
                     activity=dict(),
                     activity_category=dict(),
                     model=dict(),
                     scenario_category=dict(),
                     scenario=dict())


def _attributes_from_json(json: dict, attribute_objects: DMObjects,
                          attribute_structure: Dict[str, Tuple[Callable, str]], **kwargs) \
        -> Union[dict, Tuple[dict, NamedTuple]]:
    attributes = kwargs

    # Loop through all attributes and create them only if they were not provided by kwargs.
    for key, (func_from_json, class_name) in attribute_structure.items():
        if key not in attributes or attributes[key] is None:
            # Check if it is a list or a single object.
            if isinstance(json[key], List):  # List of objects
                attributes[key] = [_object_from_json(json_item, func_from_json, class_name,
                                                     attribute_objects)
                                   for json_item in json[key]]
            else:  # Single object
                attributes[key] = _object_from_json(json[key], func_from_json, class_name,
                                                    attribute_objects)

    return attributes


def _object_from_json(json: Union[dict, List[dict]], func_from_json: Callable, class_name: str,
                      attribute_objects: DMObjects = None, **kwargs):
    # Make sure that we have a structure to store all attributes.
    if attribute_objects is None:
        attribute_objects = get_empty_dm_object()

    # Check if we already have the object in `attribute_objects` and if so, return that one.
    if int(json["id"]) in getattr(attribute_objects, class_name):
        return getattr(attribute_objects, class_name)[int(json["id"])]

    # Run the function that returns the object.
    json_object = func_from_json(json, attribute_objects, **kwargs)  # type: ScenarioElement

    # Add object to the 'database' (i.e., `attribute_objects`).
    getattr(attribute_objects, class_name)[json_object.uid] = json_object

    # Return the object.
    return json_object
