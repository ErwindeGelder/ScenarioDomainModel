from static_environment_category import StaticEnvironmentCategory
from tags import Tag
from typing import List


class StaticEnvironment:
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
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, static_environment_category, properties=None, tags=None):
        self.name = name
        self.static_environment_category = static_environment_category  # type: StaticEnvironmentCategory
        self.properties = {} if properties is None else properties
        self.tags = [] if tags is None else tags  # Type: List[Tag]

    def get_tags(self):
        tags = self.tags
        tags += self.static_environment_category.get_tags()
        return self.tags

    def to_json(self):
        """

        :return: dictionary that can be converted to a json file
        """
        static_environment = {"name": self.name,
                              "static_environment_category": self.static_environment_category.name,
                              "property": self.properties,
                              "tag": [tag.name for tag in self.tags]}
        return static_environment
