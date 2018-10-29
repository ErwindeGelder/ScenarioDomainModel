from enum import Enum, auto


class Tag(Enum):
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
    VEH_LONG_ACT_REVERSING = auto()
    VEH_LONG_ACT_STANDING_STILL = auto()
    VEH_LONG_ACT_DRIVING_FORWARD = auto()
    VEH_LONG_ACT_DRIVING_FORWARD_BRAKING = auto()
    VEH_LONG_ACT_DRIVING_FORWARD_CRUISING = auto()
    VEH_LONG_ACT_DRIVING_FORWARD_ACCELERATING = auto()


if __name__ == "__main__":
    # List all tags
    print("List of all tags:")
    print()
    for tag in Tag:
        print(tag.name)
