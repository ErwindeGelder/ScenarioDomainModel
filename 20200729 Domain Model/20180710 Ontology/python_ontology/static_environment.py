"""
Class Activity


Author
------
Erwin de Gelder

Creation
--------
30 Oct 2018

To do
-----

Modifications
-------------

"""


from default_class import Default
from static_environment_category import StaticEnvironmentCategory
from tags import Tag
from typing import List


class StaticEnvironment(Default):
    """ Static environment

    The static environment refers to the part of a scenario that does not change during a scenario. This includes
    geo-spatially stationary elements, such as the infrastructure layout, the road layout and the type of road. Also
    the presence of buildings near the road side that act as a view-blocking obstruction are considered part of the
    static environment.
    The StaticEnvironment describes the static environment in quantitatively.

    Attributes:
        name (str): A name that serves as a short description of the static environment category.
        static_environment_category (StaticEnvironmentCategory): The category of the static environment.
        properties (dict): All properties of the static environment are stored in this dictionary.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, static_environment_category, properties=None, uid=-1, tags=None):
        # Check the types of the inputs
        if not isinstance(static_environment_category, StaticEnvironmentCategory):
            raise TypeError("Input 'static_environment_category' should be of type <StaticEnvironmentCategory> but " +
                            "is of type {0}.".format(type(static_environment_category)))
        if not isinstance(properties, dict):
            raise TypeError("Input 'properties' should be of type <dict> but is of type {0}.".format(type(properties)))

        Default.__init__(self, uid, name, tags=tags)
        self.static_environment_category = static_environment_category  # type: StaticEnvironmentCategory
        self.properties = {} if properties is None else properties

    def get_tags(self):
        tags = self.tags
        tags += self.static_environment_category.get_tags()
        return self.tags

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of StaticEnvironment to JSON.

        :return: dictionary that can be converted to a json file
        """
        static_environment = Default.to_json(self)
        static_environment["static_environment_category"] = self.static_environment_category.name
        static_environment["property"] = self.properties
        return static_environment
