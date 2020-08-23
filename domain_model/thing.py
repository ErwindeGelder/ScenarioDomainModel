""" Class Thing

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 14: When converting to JSON, ID should be a string (otherwise Cosmos DB gives error).
2018 12 06: Make it possible to return full JSON code (incl. attributes' JSON code).
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 13: Update of terminology.
2020 07 31: Return a copy of the list of tags when using get_tags().
2020 08 15: Change Default to Thing.
2020 08 22: Add function to obtain properties from a dictionary.
2020 08 23: If no uid is given, generate one.
"""

from abc import ABC, abstractmethod
from typing import List
import uuid
from .tags import Tag, tag_from_json
from .type_checking import check_for_type, check_for_list


class Thing(ABC):
    """ Thing that is used for most classes.

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
    def __init__(self, name: str = "", uid: int = None, tags: List[Tag] = None):
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
                         "tag": [tag.to_json() for tag in self.tags]}
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


def _thing_props_from_json(json: dict) -> dict:
    return dict(name=json["name"],
                uid=int(json["id"]),
                tags=[tag_from_json(tag) for tag in json["tag"]])
