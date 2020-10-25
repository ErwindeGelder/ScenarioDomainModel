""" Class ActorCategory

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 19: Enable instantiation using JSON code.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 11: Update of terminology.
2020 08 16: Make ActorCategory a subclass of DynamicPhysicalThingCategory.
2020 08 25: Add function to obtain properties from a dictionary.
2020 10 04: Change way of creating object from JSON code.
2020 10 12: ActorCategory is subclass of PhysicalElementCategory (was DynamicPhysicalThingCategory).
"""

from enum import Enum
from .physical_element_category import PhysicalElementCategory, \
    _physical_element_category_props_from_json
from .scenario_element import DMObjects, _object_from_json
from .tags import Tag
from .type_checking import check_for_type


class VehicleType(Enum):
    """ Allowed vehicle types

    The allowed vehicle types are also defines as tags in tags.py.
    """
    Vehicle = Tag.RoadUserType_Vehicle.value
    CategoryM_PassengerCar = Tag.RoadUserType_CategoryM_PassengerCar.value
    CategoryM_Minibus = Tag.RoadUserType_CategoryM_Minibus.value
    CategoryM_Bus = Tag.RoadUserType_CategoryM_Bus.value
    CategoryN_LCV = Tag.RoadUserType_CategoryN_LCV.value
    CategoryN_LGV = Tag.RoadUserType_CategoryN_LGV.value
    CategoryL_Motorcycle = Tag.RoadUserType_CategoryL_Motorcycle.value
    CategoryL_Moped = Tag.RoadUserType_CategoryL_Moped.value
    VRU_Pedestrian = Tag.RoadUserType_VRU_Pedestrian.value
    VRU_Cyclist = Tag.RoadUserType_VRU_Cyclist.value
    VRU_Other = Tag.RoadUserType_VRU_Other.value

    def to_json(self) -> dict:
        """ When tag is exporting to JSON, this function is being called

        It returns a dictionary with the "name" and the "value" of the
        VehicleType.

        :return: dictionary describing the vehicle type.
        """
        return {"name": self.name, "value": self.value}


class ActorCategory(PhysicalElementCategory):
    """ ActorCategory: Category of actor

    An actor is an agent in a scenario acting on its own behalf. "Ego vehicle"
    and "Other Road User" are types of actors in a scenario. The actor category
    only describes the actor in qualitative terms.

    Attributes:
        vehicle_type (VehicleType): The type of the actor. This should be from
            the enumeration VehicleType.
        name (str): A name that serves as a short description of the actor
            category.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        description(str): A string that qualitatively describes this actor.
    """
    def __init__(self, vehicle_type: VehicleType, **kwargs):
        # Check the types of the inputs
        check_for_type("vehicle_type", vehicle_type, VehicleType)

        PhysicalElementCategory.__init__(self, **kwargs)
        self.vehicle_type = vehicle_type  # type: VehicleType

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of ActorCategory
        to JSON.

        :return: dictionary that can be converted to a json file.
        """
        actor_category = PhysicalElementCategory.to_json(self)
        actor_category["vehicle_type"] = self.vehicle_type.to_json()
        return actor_category


def _actor_category_props_from_json(json: dict) -> dict:
    props = dict(vehicle_type=vehicle_type_from_json(json["vehicle_type"]))
    props.update(_physical_element_category_props_from_json(json))
    return props


def _actor_category_from_json(
        json: dict,
        attribute_objects: DMObjects  # pylint: disable=unused-argument
) -> ActorCategory:
    return ActorCategory(**_actor_category_props_from_json(json))


def actor_category_from_json(json: dict, attribute_objects: DMObjects = None) -> ActorCategory:
    """ Get ActorCategory object from JSON code

    It is assumed that the JSON code of the ActorCategory is created using
    ActorCategory.to_json().

    :param json: JSON code of Actor.
    :param attribute_objects: A structure for storing all objects (optional).
    :return: ActorCategory object.
    """
    return _object_from_json(json, _actor_category_from_json, "actor_category", attribute_objects)


def vehicle_type_from_json(json: dict) -> VehicleType:
    """ Get VehicleType object from JSON code.

    It is assumed that the JSON code of the VehicleType is created using
    VehicleType.to_json().

    :param json: JSON code of VehicleType.
    :return: Tag object.
    """
    return getattr(VehicleType, json["name"])
