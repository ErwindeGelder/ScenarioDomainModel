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

    # Define the tags for the carriageway user type.
    ACTOR_TYPE_CATEGORY_M = auto()
    ACTOR_TYPE_VEHICLE = auto()
    ACTOR_TYPE_PASSENGER_CAR_M1 = auto()
    ACTOR_TYPE_MINIBUS_M2 = auto()
    ACTOR_TYPE_BUS_M3 = auto()
    ACTOR_TYPE_CATEGORY_N = auto()
    ACTOR_TYPE_LCV_N1 = auto()
    ACTOR_TYPE_LGV_N2_N3 = auto()
    ACTOR_TYPE_VRU = auto()
    ACTOR_TYPE_VRU_PEDESTRIAN = auto()
    ACTOR_TYPE_VRU_CYCLIST = auto()
    ACTOR_TYPE_VRU_OTHER = auto()

    # Special tag for ego vehicle.
    EGO_VEHICLE = auto()

    # Tags for vehicle lateral and longitudinal activity.
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

    # The initial state of road users in a scenario.
    INIT_STATE_DIRECTION = auto()
    INIT_STATE_DIRECTION_SAME_AS_EGO = auto()
    INIT_STATE_DIRECTION_ONCOMING = auto()
    INIT_STATE_DIRECTION_CROSSING = auto()
    INIT_STATE_DYNAMICS = auto()
    INIT_STATE_DYNAMICS_MOVING = auto()
    INIT_STATE_DYNAMICS_STANDING_STILL = auto()
    INIT_STATE_LAT_POS = auto()
    INIT_STATE_LAT_POS_SAME_LANE = auto()
    INIT_STATE_LAT_POS_LEFT_OF_EGO = auto()
    INIT_STATE_LAT_POS_RIGHT_OF_EGO = auto()
    INIT_STATE_LONG_POS = auto()
    INIT_STATE_LONG_POS_IN_FRONT_OF_EGO = auto()
    INIT_STATE_LONG_POS_SIDE_OF_EGO = auto()
    INIT_STATE_LONG_POS_REAR_OF_EGO = auto()

    # Tags for a lead vehicle, i.e., a vehicle driving in front of ego vehicle.
    LEAD_VEHICLE_APPEARING = auto()
    LEAD_VEHICLE_APPEARING_CUTTING_IN = auto()
    LEAD_VEHICLE_APPEARING_GAP_CLOSING = auto()
    LEAD_VEHICLE_DISAPPEARING = auto()
    LEAD_VEHICLE_DISAPPEARING_CUTTING_OUT = auto()
    LEAD_VEHICLE_DISAPPEARING_GAP_OPENING = auto()
    LEAD_VEHICLE_FOLLOWING = auto()

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
