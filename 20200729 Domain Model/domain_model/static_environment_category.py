""" Class StaticEnvironmentCategory

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 21: Enable instantiation using JSON code.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 13: Update of terminology.
"""

from enum import Enum
from .default_class import Default
from .tags import Tag, tag_from_json
from .type_checking import check_for_type


class Region(Enum):
    """ Allowed regions

    The allowed regions are also defines as tags in tags.py.
    """
    EU_North = Tag.Region_EU_North.value
    EU_WestCentral = Tag.Region_EU_WestCentral.value
    EU_South = Tag.Region_EU_South.value
    USA_WestCoast = Tag.Region_USA_WestCoast.value
    USA_MidWest = Tag.Region_USA_MidWest.value
    USA_East = Tag.Region_USA_East.value
    Japan = Tag.Region_Japan.value
    China = Tag.Region_China.value
    OtherCountries = Tag.Region_OtherCountries.value

    def to_json(self) -> dict:
        """ When tag is exporting to JSON, this function is being called.

        It returns a dictionary with the "name" and the "value" of the Region.

        """
        return {"name": self.name, "value": self.value}


class StaticEnvironmentCategory(Default):
    """ Static environment category

    The static environment refers to the part of a scenario that does not change
    during a scenario. This includes geo-spatially stationary elements, such as
    the infrastructure layout, the road layout and the type of road. Also the
    presence of buildings near the road side that act as a view-blocking
    obstruction are considered part of the static environment. The
    StaticEnvironmentCategory only describes the static environment in
    qualitative terms.

    Attributes:
        description (str): A description of the static environment category.
            The objective of the description is to make the static environment
            category human interpretable.
        region(Region): Describing the region where the scenario is recorded.
        name (str): A name that serves as a short description of the static
            environment category.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, region: Region, description: str, **kwargs):
        # Check the types of the inputs.
        check_for_type("region", region, Region)
        check_for_type("description", description, str)

        Default.__init__(self, **kwargs)
        self.region = region  # type: Region
        self.description = description

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of
        StaticEnvironmentCategory to JSON.

        :return: dictionary that can be converted to a json file.
        """
        static_environment_category = Default.to_json(self)
        static_environment_category["region"] = self.region.to_json()
        static_environment_category["description"] = self.description
        return static_environment_category


def stat_env_category_from_json(json: dict) -> StaticEnvironmentCategory:
    """ Get StaticEnvironmentCategory object from JSON code

    It is assumed that the JSON code of the StaticEnvironmentCategory is created using
    StaticEnvironmentCategory.to_json().

    :param json: JSON code of StaticEnvironmentCategory.
    :return: StaticEnvironmentCategory object.
    """
    region = region_from_json(json["region"])
    stat_env = StaticEnvironmentCategory(region,
                                         json["description"],
                                         name=json["name"],
                                         uid=int(json["id"]),
                                         tags=[tag_from_json(tag) for tag in json["tag"]])
    return stat_env


def region_from_json(json: dict) -> Region:
    """ Get Region object from JSON code

    It is assumed that the JSON code of the Region is created using
    Region.to_json().

    :param json: JSON code of Region.
    :return: Tag object
    """
    return getattr(Region, json["name"])
