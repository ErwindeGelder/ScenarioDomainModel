"""
Enumeration of allowed tags in the StreetWise database.


Author
------
Erwin de Gelder

Creation
--------
29 Oct 2018

To do
-----

Modifications
-------------

"""

from enum import Enum, auto


class Tag(Enum):
    """ Tags allowed in the StreetWise database
    """

    # Define the tags for the carriageway user type
    ACTOR_TYPE_CATEGORY_M = auto()
    ACTOR_TYPE_VEHICLE = auto()
    ACTOR_TYPE_PASSENGER_CAR_M1 = auto()
    ACTOR_TYPE_MINIBUS_M2 = auto()
    ACTOR_TYPE_BUS_M3 = auto()
    ACTOR_TYPE_VRU = auto()
    ACTOR_TYPE_VRU_PEDESTRIAN = auto()
    ACTOR_TYPE_VRU_CYCLIST = auto()
    ACTOR_TYPE_VRU_OTHER = auto()

    # Special tag for ego vehicle
    EGO_VEHICLE = auto()

    # Tags for vehicle lateral and longitudinal activity
    VEH_LAT_ACT_LANE_FOLLOWING = auto()
    VEH_LAT_ACT_CHANGING_LANE = auto()
    VEH_LAT_ACT_CHANGING_LANE_LEFT = auto()
    VEH_LAT_ACT_CHANGING_LANE_RIGHT = auto()
    VEH_LAT_ACT_TURNING = auto()
    VEH_LAT_ACT_TURNING_LEFT = auto()
    VEH_LAT_ACT_TURNING_RIGHT = auto()
    VEH_LAT_ACT_SWERVING = auto()
    VEH_LAT_ACT_SWERVING_LEFT = auto()
    VEH_LAT_ACT_SWERVING_RIGHT = auto()
    VEH_LONG_ACT_REVERSING = auto()
    VEH_LONG_ACT_STANDING_STILL = auto()
    VEH_LONG_ACT_DRIVING_FORWARD = auto()
    VEH_LONG_ACT_DRIVING_FORWARD_BRAKING = auto()
    VEH_LONG_ACT_DRIVING_FORWARD_CRUISING = auto()
    VEH_LONG_ACT_DRIVING_FORWARD_ACCELERATING = auto()

    def to_json(self):
        """ When tag is exporting to JSON, this function is being called
        """
        return self.name


if __name__ == "__main__":
    # List all tags
    print("List of all tags:")
    print()
    for tag in Tag:
        print(tag.name)
