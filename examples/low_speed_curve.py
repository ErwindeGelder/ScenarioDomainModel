""" Two scenario categories with a low speed curve.

Creation date: 2019 11 12
Author(s): Erwin de Gelder

Modification:
2020 08 26 Update based on revision of the domain model.
"""

import json
import os
import domain_model as dm


def scenario_category(left: bool = False, right: bool = False) -> dm.ScenarioCategory:
    """ Create the scenario category 'low speed turn'

    :param left: Whether the bend is to the left.
    :param right: Whether the bend is to the right.
    """
    # Define the ego vehicle.
    ego = dm.ActorCategory(dm.VehicleType.Vehicle, name="Ego qualitative",
                           tags=[dm.Tag.RoadUserType_CategoryM_PassengerCar, dm.Tag.EgoVehicle,
                                 dm.Tag.VehicleLongitudinalActivity_DrivingForward,
                                 dm.Tag.VehicleLateralActivity_GoingStraight])

    # Define the static environment category.
    bend = "{:s}bend".format("left " if left else "right " if right else "")
    static_description = "Road with a {:s}".format(bend)
    road_layout = dm.StaticPhysicalThingCategory(
        description=static_description, name=bend,
        tags=[(dm.Tag.RoadLayout_Curved_Left if left else
               dm.Tag.RoadLayout_Curved_Right if right else dm.Tag.RoadLayout_Curved)]
    )

    # Define the scenario category.
    category = dm.ScenarioCategory("The ego vehicle is driving on a {:s}. To safely ".format(bend) +
                                   " and comfortably drive the bend in the road, the ego vehicle " +
                                   "needs to slow down (usually below the speed limit) on the " +
                                   "straight road preceding the curve.",
                                   os.path.join("images",
                                                "low speed curve1.pdf" if right else
                                                "low speed curve2.pdf"),
                                   name="Maneuvering a {:s}".format(bend))
    category.set_static_physical_things([road_layout])
    category.set_actors([ego])
    return category


if __name__ == "__main__":
    LEFT_BEND = scenario_category(left=True)
    RIGHT_BEND = scenario_category(right=True)
    BEND = scenario_category()
    CATEGORIES = [LEFT_BEND, RIGHT_BEND, BEND]
    NAMES = ["left bend", "right bend", "bend"]

    print("JSON code of the LEFT BEND:")
    print(json.dumps(LEFT_BEND.to_json_full(), indent=4))
    print()

    print("Demonstration of the 'includes' function:")
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            print("{:34s} {}".format("'{:s}' includes '{:s}':".format(NAMES[i], NAMES[j]),
                                     CATEGORIES[i].includes(CATEGORIES[j])))
