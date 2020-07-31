""" Enumeration of allowed tags in the StreetWise database.

Creation date: 2018 10 29
Author(s): Erwin de Gelder, Jeroen Broos

Modifications:
2018 10 13: Tags renamed according to "Scenario classes for the assessment of automated vehicles"
            by Erwin de Gelder and Olaf Op den Camp.
2018 11 08: Tags added: VehicleLateralActivity_ChangingLane & InitialState_LateralPosition_OtherLane
2018 11 12: Add tag_from_json method.
2018 11 29: Add functionality to convert Tag to a Str (string).
2018 12 07: Many "supertags" added.
2018 12 07: Functionality for determining whether a Tag is a subtag of another Tag.
2018 12 10: Region tags added.
2019 11 04: Add str() around tag.name as to avoid Pylint from complaining.
2020 07 31: Change is_subtag to is_supertag_of to make it more descriptive.
"""

from enum import Enum, unique


# this is for compatibility with python3.5 in Azure, the unique decorator ensures enumerations
# get unique numbers
@unique
class Tag(Enum):
    """ Tags allowed in the StreetWise database
    """

    # Define the tags for the carriageway user type.
    RoadUserType_Vehicle = 1
    RoadUserType_CategoryM_PassengerCar = 2
    RoadUserType_CategoryM_Bus = 3
    RoadUserType_CategoryM_Minibus = 4
    RoadUserType_CategoryN_LCV = 5
    RoadUserType_CategoryN_LGV = 6
    RoadUserType_CategoryL_Motorcycle = 7
    RoadUserType_CategoryL_Moped = 8
    RoadUserType_VRU = 9
    RoadUserType_VRU_Pedestrian = 10
    RoadUserType_VRU_Cyclist = 11
    RoadUserType_VRU_Other = 12

    # Special tag for ego vehicle.
    EgoVehicle = 13

    # Tags for vehicle lateral and longitudinal activity.
    VehicleLateralActivity_GoingStraight = 14
    VehicleLateralActivity_ChangingLane = 15
    VehicleLateralActivity_ChangingLane_Left = 16
    VehicleLateralActivity_ChangingLane_Right = 17
    VehicleLateralActivity_Turning = 120
    VehicleLateralActivity_Turning_Left = 18
    VehicleLateralActivity_Turning_Right = 19
    VehicleLateralActivity_Swerving = 121
    VehicleLateralActivity_Swerving_Left = 20
    VehicleLateralActivity_Swerving_Right = 21
    VehicleLongitudinalActivity_Reversing = 22
    VehicleLongitudinalActivity_StandingStill = 23
    VehicleLongitudinalActivity_DrivingForward = 24
    VehicleLongitudinalActivity_DrivingForward_Braking = 25
    VehicleLongitudinalActivity_DrivingForward_Cruising = 26
    VehicleLongitudinalActivity_DrivingForward_Accelerating = 27

    # Tags for pedestrian activities
    PedestrianActivity_Walking = 122
    PedestrianActivity_Walking_Straight = 28
    PedestrianActivity_Walking_TurnLeft = 29
    PedestrianActivity_Walking_TurnRight = 30
    PedestrianActivity_Stopping = 31
    PedestrianActivity_StandingStill = 32

    # Tags for cyclist activities
    CyclistLateralActivity_GoingStraight = 33
    CyclistLateralActivity_Turning = 123
    CyclistLateralActivity_Turning_Left = 34
    CyclistLateralActivity_Turning_Right = 35
    CyclistLateralActivity_Swerving = 124
    CyclistLateralActivity_Swerving_Left = 36
    CyclistLateralActivity_Swerving_Right = 37
    CyclistLongitudinalActivity_RidingForward = 125
    CyclistLongitudinalActivity_RidingForward_Decelerating = 38
    CyclistLongitudinalActivity_RidingForward_Cruising = 39
    CyclistLongitudinalActivity_RidingForward_Accelerating = 40
    CyclistLongitudinalActivity_Stopping = 41
    CyclistLongitudinalActivity_StandingStill = 42

    # The initial state of road users in a scenario.
    InitialState_Direction_SameAsEgo = 43
    InitialState_Direction_Oncoming = 44
    InitialState_Direction_Crossing = 45
    InitialState_Dynamics_Moving = 46
    InitialState_Dynamics_StandingStill = 47
    InitialState_LateralPosition_SameLane = 48
    InitialState_LateralPosition_LeftOfEgo = 49
    InitialState_LateralPosition_RightOfEgo = 50
    InitialState_LateralPosition_OtherLane = 51
    InitialState_LongitudinalPosition_InFrontOfEgo = 52
    InitialState_LongitudinalPosition_SideOfEgo = 53
    InitialState_LongitudinalPosition_RearOfEgo = 54

    # Tags for a lead vehicle, i.e., a vehicle driving in front of ego vehicle.
    LeadVehicle_Appearing = 126
    LeadVehicle_Appearing_CuttingIn = 55
    LeadVehicle_Appearing_GapClosing = 56
    LeadVehicle_Disappearing = 127
    LeadVehicle_Disappearing_CuttingOut = 57
    LeadVehicle_Disappearing_GapOpening = 58
    LeadVehicle_Following = 59

    # Tags that describe the presence of animals
    Animal_Position_OnEgoPath = 60
    Animal_Position_OnRoad = 61
    Animal_Position_NextToRoad = 62
    Animal_Dynamics_Moving = 63
    Animal_Dynamics_Stationary = 64

    # Tags that describe the type of road
    RoadType_PrincipleRoad = 128
    RoadType_PrincipleRoad_Motorway = 65
    RoadType_PrincipleRoad_Trunk = 66
    RoadType_PrincipleRoad_Primary = 67
    RoadType_PrincipleRoad_Secondary = 68
    RoadType_PrincipleRoad_Tertiary = 69
    RoadType_PrincipleRoad_Unclassified = 70
    RoadType_PrincipleRoad_Residential = 71
    RoadType_PrincipleRoad_Service = 72
    RoadType_Link = 129
    RoadType_Link_MotorwayLink = 73
    RoadType_Link_TrunkLink = 74
    RoadType_Link_PrimaryLink = 75
    RoadType_Link_SecondaryLink = 76
    RoadType_Link_TertiaryLink = 77
    RoadType_Link_Sliproad = 78
    RoadType_Pavement = 130
    RoadType_Pavement_Footway = 79
    RoadType_Pavement_CyclistPath = 80

    # Tags that describe the road layout
    RoadLayout_Straight = 131
    RoadLayout_Straight_Merge = 81
    RoadLayout_Straight_Entrance = 82
    RoadLayout_Straight_Exit = 83
    RoadLayout_Straight_Other = 84
    RoadLayout_Curved = 132
    RoadLayout_Curved_Merge = 85
    RoadLayout_Curved_Entrance = 86
    RoadLayout_Curved_Exit = 87
    RoadLayout_Curved_Other = 88
    RoadLayout_Curved_Left = 152
    RoadLayout_Curved_Right = 153
    RoadLayout_Junction = 133
    RoadLayout_Junction_TrafficLight = 89
    RoadLayout_Junction_NoTrafficLight = 90
    RoadLayout_PedestrianCrossing = 134
    RoadLayout_PedestrianCrossing_TrafficLight = 91
    RoadLayout_PedestrianCrossing_NoTrafficLight = 92

    # Tags that describe the region
    Region_EU_North = 143
    Region_EU_WestCentral = 144
    Region_EU_South = 145
    Region_USA_WestCoast = 146
    Region_USA_MidWest = 147
    Region_USA_East = 148
    Region_Japan = 149
    Region_China = 150
    Region_OtherCountries = 151

    # Tags that describe a static object
    StaticObject_OnIntendedPath = 135
    StaticObject_OnIntendedPath_Passable = 93
    StaticObject_OnIntendedPath_Impassable = 94
    StaticObject_ViewBlocking = 95
    StaticObject_Other = 96

    # Tags that describe the traffic light status for the ego vehicle
    TrafficLight_Red = 97
    TrafficLight_Amber = 98
    TrafficLight_Green = 99
    TrafficLight_NA = 100

    # Tags for weather and lighting conditions
    Weather_NoPrecipitation = 101
    Weather_Rain = 136
    Weather_Rain_Light = 102
    Weather_Rain_Moderate = 103
    Weather_Rain_Heavy = 104
    Weather_Suspension = 137
    Weather_Suspension_Mist = 105
    Weather_Suspension_Fog = 106
    Weather_Suspension_Haze = 107
    Weather_Snow = 138
    Weather_Snow_Light = 108
    Weather_Snow_Moderate = 109
    Weather_Snow_Heavy = 110
    Lighting_Day = 139
    Lighting_Day_ClearSky = 111
    Lighting_Day_Cloudy = 112
    Lighting_Day_Overcast = 113
    Lighting_Twilight = 140
    Lighting_Twilight_Dawn = 114
    Lighting_Twilight_Dusk = 115
    Lighting_Dark = 141
    Lighting_Dark_Streetlights = 116
    Lighting_Dark_NoStreetlights = 117
    Lighting_Glare = 142
    Lighting_Glare_Sun = 118
    Lighting_Glare_OncomingTraffic = 119

    def to_json(self) -> str:
        """ When tag is exporting to JSON, this function is being called

        This function simply returns the name of the Tag.

        :return: Name of the Tag.
        """
        return self.name

    def __str__(self):
        return self.name

    def is_supertag_of(self, subtag) -> bool:
        """ Check whether the provided Tag is a subtag of the current Tag.

        :param subtag: The potential subtag.
        :return: True if the provided Tag is a "subtag", otherwise False.
        """
        # If the subtag is the same, then it is regarding to also be a subtag.
        if self == subtag:
            return True

        # If the subtag is exactly the same, but with more details, it is regarded to be a
        # subtag. For example,
        if len(subtag.name) > len(self.name) and subtag.name[:len(self.name)] == str(self.name):
            return True

        # In case the own Tag is a "RoadUserType_Vehicle", then there are many other
        # RoadUserTypes that are regarded as "subtags" of "Vehicle".
        if self == Tag.RoadUserType_Vehicle:
            vehicle_tags = [Tag.RoadUserType_CategoryM_PassengerCar,
                            Tag.RoadUserType_CategoryM_Bus,
                            Tag.RoadUserType_CategoryM_Minibus,
                            Tag.RoadUserType_CategoryN_LCV,
                            Tag.RoadUserType_CategoryN_LGV,
                            Tag.RoadUserType_CategoryL_Motorcycle,
                            Tag.RoadUserType_CategoryL_Moped]
            if subtag in vehicle_tags:
                return True

        return False


def tag_from_json(json: str) -> Tag:
    """ Get Tag object from JSON code

    It is assumed that the JSON code of the Tag is created using Tag.to_json().

    :param json: JSON code of Tag, which is simply a string of the name of the Tag.
    :return: Tag object.
    """

    return getattr(Tag, json)


if __name__ == "__main__":
    # List all tags
    print("List of all tags:")
    print()
    for tag in Tag:
        print(tag.name)
