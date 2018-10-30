"""
Class ActorCategory


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


from default_class import DefaultClass
from tags import Tag
from typing import List
from enum import Enum


class ActorCategory(DefaultClass):
    """ ActorCategory: Category of actor

    An actor is an agent in a scenario acting on its own behalf. "Ego vehicle" and "Other Road User" are types of
    actors in a scenario. The actor category only describes the actor in qualitative terms.

    Attributes:
        name (str): A name that serves as a short description of the actor category.
        vehicle_type (VehicleType): The type of the actor. This should be from the enumeration VehicleType.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, vehicle_type, tags=None):
        # Check the types of the inputs
        if not isinstance(vehicle_type, VehicleType):
            raise TypeError("Input 'vehicle_type' should be of type <VehicleType> but is of type {0}.".
                            format(type(vehicle_type)))

        DefaultClass.__init__(self, name, tags=tags)
        self.vehicle_type = vehicle_type  # type: VehicleType

    def get_tags(self):
        return self.tags

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of ActorCategory to JSON.

        :return: dictionary that can be converted to a json file
        """
        actor_category = DefaultClass.to_json(self)
        actor_category["vehicle_type"] = self.vehicle_type.to_json()
        return actor_category


class VehicleType(Enum):
    CATEGORY_M = Tag.ACTOR_TYPE_CATEGORY_M.value
    VEHICLE = Tag.ACTOR_TYPE_VEHICLE.value
    PASSENGER_CAR_M1 = Tag.ACTOR_TYPE_PASSENGER_CAR_M1.value
    MINIBUS_M2 = Tag.ACTOR_TYPE_MINIBUS_M2.value
    BUS_M3 = Tag.ACTOR_TYPE_BUS_M3.value
    VRU = Tag.ACTOR_TYPE_VRU.value
    PEDESTRIAN = Tag.ACTOR_TYPE_VRU_PEDESTRIAN.value
    VRU_CYCLIST = Tag.ACTOR_TYPE_VRU_CYCLIST.value
    VRU_OTHER = Tag.ACTOR_TYPE_VRU_OTHER.value

    def to_json(self):
        """ When tag is exporting to JSON, this function is being called
        """
        return {"name": self.name, "value": self.value}
