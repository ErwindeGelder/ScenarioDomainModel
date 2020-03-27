"""
Class StaticEnvironment


Author
------
Erwin de Gelder, Jeroen Broos

Creation
--------
30 Oct 2018

To do
-----

Modifications
-------------
05 Nov 2018: Make code PEP8 compliant.
22 Nov 2018: Make it possible to instantiate StaticEnvironment from JSON code.
06 Dec 2018: Make it possible to return full JSON code (incl. attributes' JSON code).
15 Jan 2019: to_opendrive function added
22 May 2019: Make use of type_checking.py to shorten the initialization.
13 Oct 2019: Update of terminology.
27 Mar 2020: Add the possibility to pass on the static environment category for the to_json().
"""

from typing import List
from .default_class import Default
from .static_environment_category import StaticEnvironmentCategory, stat_env_category_from_json
from .tags import Tag, tag_from_json
from .type_checking import check_for_type


class StaticEnvironment(Default):
    """ Static environment

    The static environment refers to the part of a scenario that does not change
    during a scenario. This includes geo-spatially stationary elements, such as
    the infrastructure layout, the road layout and the type of road. Also the
    presence of buildings near the road side that act as a view-blocking
    obstruction are considered part of the static environment. The
    StaticEnvironment describes the static environment in quantitatively.

    Attributes:
        static_environment_category (StaticEnvironmentCategory): The category of
            the static environment.
        properties (dict): All properties of the static environment are stored
            in this dictionary.
        name (str): A name that serves as a short description of the static
            environment category.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, static_environment_category: StaticEnvironmentCategory,
                 properties: dict = None, **kwargs):
        # Check the types of the inputs
        check_for_type("static_environment_category", static_environment_category,
                       StaticEnvironmentCategory)
        if properties is not None:
            check_for_type("properties", properties, dict)

        Default.__init__(self, **kwargs)
        self.static_environment_category = static_environment_category
        self.properties = {} if properties is None else properties

    def get_tags(self) -> List[Tag]:
        """ Return the list of tags related to this StaticEnvironment.

        It returns the tags associated to this Activity and the tags associated
        with the StaticEnvironmentCategory.

        :return: List of tags.
        """
        tags = self.tags
        tags += self.static_environment_category.get_tags()
        return self.tags

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of
        StaticEnvironment to JSON.

        :return: dictionary that can be converted to a json file.
        """
        static_environment = Default.to_json(self)
        static_environment["static_environment_category"] = \
            {"name": self.static_environment_category.name,
             "uid": self.static_environment_category.uid}
        static_environment["property"] = self.properties
        return static_environment

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
        static_environment = self.to_json()
        static_environment["static_environment_category"] = \
            self.static_environment_category.to_json_full()
        return static_environment


def stat_env_from_json(json: dict, stat_env_category: StaticEnvironmentCategory = None) \
        -> StaticEnvironment:
    """ Get StaticEnvironment object from JSON code

    It is assumed that all the attributes are fully defined. Hence,
    the StaticEnvironmentCategory need to be fully defined instead of only the
    unique ID. Alternatively, the StaticEnvironmentCategory can be passed as
    optional argument. In that case, the StaticEnvironmentCategory does not need
    to be defined in the JSON code.

    :param json: JSON code of StaticEnvironment.
    :param stat_env_category: If given, it will not be based on the JSON code.
    :return: StaticEnvironment object.
    """
    if stat_env_category is None:
        stat_env_category = stat_env_category_from_json(json["static_environment_category"])
    stat_env = StaticEnvironment(stat_env_category,
                                 properties=json["property"],
                                 name=json["name"],
                                 uid=int(json["id"]),
                                 tags=[tag_from_json(tag) for tag in json["tag"]])
    return stat_env
