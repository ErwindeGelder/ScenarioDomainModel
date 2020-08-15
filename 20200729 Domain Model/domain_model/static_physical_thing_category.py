""" Class StaticPhysicalThingCategory

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
"""

from .physical_thing_category import PhysicalThingCategory
from .tags import tag_from_json


class StaticPhysicalThingCategory(PhysicalThingCategory):
    """ Static Physical Thing Category

    The static environment refers to the part of a scenario that does not change
    during a scenario. This includes geo-spatially stationary elements, such as
    the infrastructure layout, the road layout and the type of road. Also the
    presence of buildings near the road side that act as a view-blocking
    obstruction are considered part of the static environment. The
    StaticPhysicalThingCategory describes the physical things that are part of
    the static environment in a qualitative manner.

    Attributes:
        description (str): A description of the static physical thing category.
            The objective of the description is to make the static physical
            thing category human interpretable.
        name (str): A name that serves as a short description of the static
            environment category.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, description: str = "", **kwargs):
        PhysicalThingCategory.__init__(self, description=description, **kwargs)


def static_physical_thing_category_from_json(json: dict) -> StaticPhysicalThingCategory:
    """ Get StaticPhysicalThingCategory object from JSON code

    It is assumed that the JSON code of the StaticPhysicalThingCategory is
    created using StaticPhysicalThingCategory.to_json().

    :param json: JSON code of StaticEnvironmentCategory.
    :return: StaticPhysicalThingCategory object.
    """
    stat_env = StaticPhysicalThingCategory(description=json["description"],
                                           name=json["name"],
                                           uid=int(json["id"]),
                                           tags=[tag_from_json(tag) for tag in json["tag"]])
    return stat_env
