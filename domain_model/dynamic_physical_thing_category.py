""" Class DynamicPhysicalThingCategory

Creation date: 2020 08 16
Author(s): Erwin de Gelder

Modifications:
2020 08 25: Add function to obtain properties from a dictionary.
"""

from .physical_thing_category import PhysicalThingCategory, _physical_thing_category_props_from_json


class DynamicPhysicalThingCategory(PhysicalThingCategory):
    """ Dynamic Physical Thing Category

    A DynamicPhysicalThingCategory is used to qualitatively describe a dynamic
    physical thing. A dynamic thing is "dynamic" if it changes during the
    scenario.

    Attributes:
        description (str): A description of the dynamic physical thing category.
            The objective of the description is to make the dynamic physical
            thing category human interpretable.
        name (str): A name that serves as a short description of the static
            environment category.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, description: str = "", **kwargs):
        PhysicalThingCategory.__init__(self, description=description, **kwargs)


def _dynamic_physical_thing_category_props_from_json(json: dict) -> dict:
    return _physical_thing_category_props_from_json(json)


def dynamic_physical_thing_category_from_json(json: dict) -> DynamicPhysicalThingCategory:
    """ Get DynamicPhysicalThingCategory object from JSON code

    It is assumed that the JSON code of the DynamicPhysicalThingCategory is
    created using DynamicPhysicalThingCategory.to_json().

    :param json: JSON code of DynamicPhysicalThingCategory.
    :return: DynamicPhysicalThingCategory object.
    """
    return DynamicPhysicalThingCategory(**_dynamic_physical_thing_category_props_from_json(json))
