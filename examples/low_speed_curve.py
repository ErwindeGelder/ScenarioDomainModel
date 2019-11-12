"""
Two scenario categories with a low speed curve.

Author
------
Erwin de Gelder

Creation
--------
12 Nov 2019

To do
-----
Demonstrate the "includes" method of ScenarioCategory.

Modification
------------
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
    crossing = dm.StaticEnvironmentCategory(
        region=dm.Region.OtherCountries, description=static_description, name=bend,
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
                                   crossing,
                                   name="Maneuvering a {:s}".format(bend))
    category.set_actors([ego], update_uids=True)
    return category


if __name__ == "__main__":
    LEFT_BEND = scenario_category(left=False)
    print(json.dumps(LEFT_BEND.to_json_full(), indent=4))
