"""
Example of a scenario category and scenario from the article.

For more details on this example, see the following article:
E. de Gelder, J.-P. Paardekooper, A. Khabbaz Saberi, H. Elrofai, O. Op den Camp,
J. Ploeg, L. Friedmann, and B. De Schutter, "Ontology for Scenarios for the
Assessment of Automated Vehicle", 2019, To be published.

Author
------
Erwin de Gelder

Creation
--------
05 May 2019

To do
-----

Modification
------------
04 Nov 2019 Update based on updates of the domain model.

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
    crossing = dm.StaticEnvironmentCategory(
        region=dm.Region.EU_WestCentral, description=static_description,
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
        os.path.join("images", "pedestrian_crossing.pdf"), crossing,
        name="Pedestrian crossing"
    )
    category.set_actors([ego, pedestrian], update_uids=True)
    category.set_activities([braking, stationary, accelerating, walking], update_uids=True)
    category.set_acts([(ego, braking), (ego, stationary), (ego, accelerating),
                       (pedestrian, walking)])
    return category


def scenario() -> dm.Scenario:
    """ Create a scenario "pedestrian crossing".

    :return: The Scenario "pedestrian crossing".
    """
    category = scenario_category()

    # Define the actors.
    ego = dm.EgoVehicle(category.actor_categories[0],
                        initial_states=[dm.State(dm.StateVariable.LONGITUDINAL_POSITION, -20),
                                        dm.State(dm.StateVariable.LATERAL_POSITION, -1.5),
                                        dm.State(dm.StateVariable.HEADING, np.pi / 2)],
                        name="Ego")
    pedestrian = dm.Actor(category.actor_categories[1],
                          initial_states=[dm.State(dm.StateVariable.LONGITUDINAL_POSITION, 0),
                                          dm.State(dm.StateVariable.HEADING, 0)],
                          name="Pedestrian")

    # Define the static environment.
    static = dm.StaticEnvironment(category.static_environment,
                                  properties=dict(road=dict(lanes=2, lanewidth=3,
                                                            xy=[(-60, 0), (60, 0)]),
                                                  footway=dict(width=3,
                                                               xy=[(0, 6), (0, -6)])),
                                  name="Pedestrian crossing")

    # Define the activities.
    braking = dm.SetActivity(category.activity_categories[0], 0, 4, dict(xstart=8, xend=0),
                             name="Ego braking")
    stationary = dm.SetActivity(category.activity_categories[1], 4, 3, dict(xstart=0),
                                name="Ego stationary")
    accelerating = dm.SetActivity(category.activity_categories[2], 7, 5,
                                  dict(xstart=0, xend=7.5), name="Ego accelerating")
    walking = dm.SetActivity(category.activity_categories[3], 0, 12, dict(xstart=-6, xend=6),
                             name="Pedestrian walking")

    # Define the scenario.
    scen = dm.Scenario(0, 12, static, name="Ego braking for crossing pedestrian")
    scen.set_actors([ego, pedestrian], update_uids=True)
    scen.set_activities([braking, stationary, accelerating, walking], update_uids=True)
    scen.set_acts([(ego, braking, 0), (ego, stationary, 4), (ego, accelerating, 7),
                   (pedestrian, walking, 0)])
    return scen


if __name__ == "__main__":
    print(json.dumps(scenario().to_json_full(), indent=4))
