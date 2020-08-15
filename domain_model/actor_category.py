""" Class ActorCategory

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 19: Enable instantiation using JSON code.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 11: Update of terminology.
"""

from enum import Enum
from .tags import Tag, tag_from_json
from .thing import Thing
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


class ActorCategory(Thing):
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
    """
    def __init__(self, vehicle_type: VehicleType, **kwargs):
        # Check the types of the inputs
        check_for_type("vehicle_type", vehicle_type, VehicleType)

        Thing.__init__(self, **kwargs)
        self.vehicle_type = vehicle_type  # type: VehicleType

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of ActorCategory
        to JSON.

        :return: dictionary that can be converted to a json file.
        """
        actor_category = Thing.to_json(self)
        actor_category["vehicle_type"] = self.vehicle_type.to_json()
        return actor_category


def actor_category_from_json(json: dict) -> ActorCategory:
    """ Get ActorCategory object from JSON code

    It is assumed that the JSON code of the ActorCategory is created using
    ActorCategory.to_json().

    :param json: JSON code of Actor.
    :return: ActorCategory object.
    """
    vehicle_type = vehicle_type_from_json(json["vehicle_type"])
    actor_category = ActorCategory(vehicle_type, name=json["name"], uid=int(json["id"]),
                                   tags=[tag_from_json(tag) for tag in json["tag"]])
    return actor_category


def vehicle_type_from_json(json: dict) -> VehicleType:
    """ Get VehicleType object from JSON code.

    It is assumed that the JSON code of the VehicleType is created using
    VehicleType.to_json().

    :param json: JSON code of VehicleType.
    :return: Tag object.
    """

    return getattr(VehicleType, json["name"])
