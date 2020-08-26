""" Example of a scenario category and scenario from the article.

Creation data: 2019 05 05
Author(s): Erwin de Gelder

Modifications:
2019 11 04 Update based on updates of the domain model.
2020 08 26 Update based on revision of the domain model.
"""

import json
import os
import numpy as np
import domain_model as dm


def scenario_category() -> dm.ScenarioCategory:
    """ Create the scenario category "pedestrian crossing".

    :return: The ScenarioCategory "pedestrian crossing".
    """
    # Define the actor categories.
    ego = dm.ActorCategory(dm.VehicleType.Vehicle, name="Ego qualitative",
                           tags=[dm.Tag.RoadUserType_CategoryM_PassengerCar, dm.Tag.EgoVehicle])
    pedestrian = dm.ActorCategory(dm.VehicleType.VRU_Pedestrian, name="Pedestrian qualitative",
                                  tags=[dm.Tag.RoadUserType_VRU_Pedestrian])

    # Define the static environment category.
    static_description = "Straight road with two lanes and a pedestrian crossing"
    crossing = dm.StaticPhysicalThingCategory(
        description=static_description,
        name="Pedestrian crossing qualitative",
        tags=[dm.Tag.RoadLayout_PedestrianCrossing_NoTrafficLight]
    )

    # Define the activity categories.
    braking = dm.ActivityCategory(
        dm.Sinusoidal(), dm.StateVariable.SPEED, name="Braking",
        tags=[dm.Tag.VehicleLongitudinalActivity_DrivingForward_Braking]
    )
    stationary = dm.ActivityCategory(
        dm.Constant(), dm.StateVariable.SPEED, name="Stationary",
        tags=[dm.Tag.VehicleLongitudinalActivity_StandingStill]
    )
    accelerating = dm.ActivityCategory(
        dm.Linear(), dm.StateVariable.SPEED, name="Accelerating",
        tags=[dm.Tag.VehicleLongitudinalActivity_DrivingForward_Accelerating]
    )
    walking = dm.ActivityCategory(
        dm.Linear(), dm.StateVariable.LATERAL_POSITION, name="Walking straight",
        tags=[dm.Tag.PedestrianActivity_Walking_Straight]
    )

    # Define the scenario category.
    category = dm.ScenarioCategory(
        "Straight road with two lanes and a pedestrian crossing",
        os.path.join("images", "pedestrian_crossing.pdf"),
        name="Pedestrian crossing"
    )
    category.set_static_physical_things([crossing])
    category.set_actors([ego, pedestrian])
    category.set_activities([braking, stationary, accelerating, walking])
    category.set_acts([(ego, braking), (ego, stationary), (ego, accelerating),
                       (pedestrian, walking)])
    return category


def scenario() -> dm.Scenario:
    """ Create a scenario "pedestrian crossing".

    :return: The Scenario "pedestrian crossing".
    """
    category = scenario_category()

    # Define the actors.
    ego = dm.EgoVehicle(category.actors[0],
                        initial_states=[dm.State(dm.StateVariable.LONGITUDINAL_POSITION, -20),
                                        dm.State(dm.StateVariable.LATERAL_POSITION, -1.5),
                                        dm.State(dm.StateVariable.HEADING, np.pi / 2)],
                        name="Ego")
    pedestrian = dm.Actor(category.actors[1],
                          initial_states=[dm.State(dm.StateVariable.LONGITUDINAL_POSITION, 0),
                                          dm.State(dm.StateVariable.HEADING, 0)],
                          name="Pedestrian")

    # Define the static environment.
    static = dm.StaticPhysicalThing(category.static_physical_things[0],
                                    properties=dict(road=dict(lanes=2, lanewidth=3,
                                                              xy=[(-60, 0), (60, 0)]),
                                                    footway=dict(width=3,
                                                                 xy=[(0, 6), (0, -6)])),
                                    name="Pedestrian crossing")

    # Define the activities.
    braking = dm.Activity(category.activities[0], dict(xstart=8, xend=0), start=0, end=4,
                          name="Ego braking")
    stationary = dm.Activity(category.activities[1], dict(xstart=0), start=braking.end, end=7,
                             name="Ego stationary")
    accelerating = dm.Activity(category.activities[2], dict(xstart=0, xend=7.5),
                               start=stationary.end, end=12, name="Ego accelerating")
    walking = dm.Activity(category.activities[3], dict(xstart=-6, xend=6), start=braking.start,
                          end=accelerating.end, name="Pedestrian walking")

    # Define the scenario.
    scen = dm.Scenario(start=braking.start, end=accelerating.end,
                       name="Ego braking for crossing pedestrian")
    scen.set_static_physical_things([static])
    scen.set_actors([ego, pedestrian])
    scen.set_activities([braking, stationary, accelerating, walking])
    scen.set_acts([(ego, braking), (ego, stationary), (ego, accelerating), (pedestrian, walking)])
    return scen


if __name__ == "__main__":
    CATEGORY = scenario_category()
    SCENARIO = scenario()

    print("JSON code of the scenario:")
    print(json.dumps(SCENARIO.to_json_full(), indent=4))
    print()
    print("JSON code of the scenario category:")
    print(json.dumps(CATEGORY.to_json_full(), indent=4))
    print()
    print("Does the scenario category comprise the scenario?")
    print(CATEGORY.comprises(SCENARIO))
