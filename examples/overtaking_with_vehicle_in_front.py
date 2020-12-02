""" Example of a scenario with another vehicle overtaking the ego vehicle while
there is a vehicle in front of the ego vehicle.

Creation date: 2020 12 02
Author(s): Erwin de Gelder

Modifications:
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import domain_model as dm


def scenario_category() -> dm.ScenarioCategory:
    """ Create the scenario category "overtaking with vehicle in front".

    :return: The ScenarioCategory "overtaking with vehicle in front".
    """
    # Define the actor categories.
    ego = dm.ActorCategory(dm.VehicleType.Vehicle, name="Ego qualitative",
                           tags=[dm.Tag.RoadUserType_CategoryM_PassengerCar,
                                 dm.Tag.EgoVehicle])
    lead = dm.ActorCategory(dm.VehicleType.Vehicle, name="Lead vehicle",
                            tags=[dm.Tag.RoadUserType_Vehicle])
    overtaking = dm.ActorCategory(dm.VehicleType.Vehicle, name="Overtaking vehicle",
                                  tags=[dm.Tag.RoadUserType_Vehicle,
                                        dm.Tag.InitialState_LongitudinalPosition_RearOfEgo])

    # Define the static physical thing category.
    static_description = "Straight road with two lanes"
    highway = dm.PhysicalElementCategory(
        description=static_description,
        name="Highway with 2 lanes",
        tags=[dm.Tag.RoadLayout_Straight]
    )

    # Define the activity categories.
    splines = dm.Splines()
    longitudinal = dm.ActivityCategory(
        splines, dm.StateVariable.SPEED, name="driving forward",
        tags=[dm.Tag.VehicleLongitudinalActivity_DrivingForward]
    )
    lateral = dm.ActivityCategory(
        splines, dm.StateVariable.LATERAL_POSITION, name="lateral position",
        tags=[dm.Tag.VehicleLateralActivity_GoingStraight]
    )

    # Define the scenario category.
    category = dm.ScenarioCategory("", "Overtaking with vehicle in front",
                                   name="Overtaking with vehicle in front")
    category.set_physical_elements([highway])
    category.set_actors([ego, lead, overtaking])
    category.set_activities([longitudinal, lateral])
    category.set_acts([(lead, longitudinal), (lead, lateral),
                       (overtaking, longitudinal), (overtaking, lateral)])
    return category


def scenario() -> dm.Scenario:
    """ Create a scenario "overtaking with vehicle in front".

    :return: The Scenario "overtaking with vehicle in front".
    """
    category = scenario_category()

    # Define the actors.
    ego = dm.EgoVehicle(category.actors[0],
                        initial_states=[dm.State(dm.StateVariable.LONGITUDINAL_POSITION, 0),
                                        dm.State(dm.StateVariable.LATERAL_POSITION, 0),
                                        dm.State(dm.StateVariable.SPEED, 15)],
                        name="Ego vehicle")
    lead = dm.EgoVehicle(category.actors[1],
                         initial_states=[dm.State(dm.StateVariable.LONGITUDINAL_POSITION, 20),
                                         dm.State(dm.StateVariable.LATERAL_POSITION, 0)],
                         name="Lead vehicle")
    overtaking = dm.EgoVehicle(category.actors[2],
                               initial_states=[dm.State(dm.StateVariable.LONGITUDINAL_POSITION,
                                                        -20),
                                               dm.State(dm.StateVariable.LATERAL_POSITION, 3.5)],
                               name="Overtaking vehicle")

    # Define the static environment.
    static = dm.PhysicalElement(category.physical_elements[0],
                                properties=dict(road=dict(lanes=2, lanewidth=3,
                                                          xy=[(-100, 0), (500, 0)])),
                                name="Highway straight")

    # Define the activities.
    speed_overtaking = dm.Activity(category.activities[0],
                                   dict(knots=[0, 0, 0, 0, .25, .5, .75, 1, 1, 1, 1],
                                        coefficients=[20, 20, 22, 20.5, 19.5, 20, 20],
                                        degree=3),
                                   start=0, end=10)
    lat_overtaking = dm.Activity(category.activities[1],
                                 dict(knots=[0, 0, 0, 0, .25, .5, .75, 1, 1, 1, 1],
                                      coefficients=[3.5, 3.5, 3.2, 3.4, 3.6, 3.7, 3.7],
                                      degree=3),
                                 start=speed_overtaking.start, end=speed_overtaking.end)
    speed_lead = dm.Activity(category.activities[0],
                             dict(knots=[0, 0, 0, 0, .25, .5, .75, 1, 1, 1, 1],
                                  coefficients=[15.4, 15.4, 16, 15, 15.5, 15.8, 15.8],
                                  degree=3),
                             start=speed_overtaking.start, end=speed_overtaking.end)
    lat_lead = dm.Activity(category.activities[1],
                           dict(knots=[0, 0, 0, 0, .25, .5, .75, 1, 1, 1, 1],
                                coefficients=[-.1, -.1, -.2, 0, .1, .2, .2],
                                degree=3),
                           start=speed_overtaking.start, end=speed_overtaking.end)

    # Define the scenario.
    scen = dm.Scenario(start=speed_overtaking.start, end=speed_overtaking.end,
                       name="Overtaking vehicle with vehicle in front")
    scen.set_physical_elements([static])
    scen.set_actors([ego, lead, overtaking])
    scen.set_activities([speed_overtaking, lat_overtaking, speed_lead, lat_lead])
    scen.set_acts([(overtaking, speed_overtaking), (overtaking, lat_overtaking),
                   (lead, speed_lead), (lead, lat_lead)])
    return scen


if __name__ == "__main__":
    CATEGORY = scenario_category()
    print(CATEGORY)
    SCENARIO = scenario()

    # Store the scenario in a .json file.
    FILENAME = "overtaking_with_vehicle_in_front.json"
    with open(FILENAME, "w") as FILE:
        json.dump(SCENARIO.to_json_full(), FILE, indent=4)

    # Obtain the secnario from the .json file. This should give the same result.
    with open(FILENAME, "r") as FILE:
        SCENARIO = dm.scenario_from_json(json.load(FILE))

    TIME = np.linspace(SCENARIO.get_tstart(), SCENARIO.get_tend(), 50)
    _, (AX1, AX2) = plt.subplots(1, 2, figsize=(10, 5))
    SPEED = SCENARIO.get_state(SCENARIO.get_actor_by_name("Overtaking vehicle"),
                               dm.StateVariable.SPEED, TIME)
    AX1.plot(TIME, SPEED, label="Overtaking vehicle")
    SPEED = SCENARIO.get_state(SCENARIO.get_actor_by_name("Lead vehicle"),
                               dm.StateVariable.SPEED, TIME)
    AX1.plot(TIME, SPEED, label="Lead vehicle")
    AX1.set_xlabel("Time [s]")
    AX1.set_ylabel("Speed [m/s]")
    AX1.legend()

    LAT = SCENARIO.get_state(SCENARIO.get_actor_by_name("Overtaking vehicle"),
                             dm.StateVariable.LATERAL_POSITION, TIME)
    AX2.plot(TIME, LAT, label="Overtaking vehicle")
    LAT = SCENARIO.get_state(SCENARIO.get_actor_by_name("Lead vehicle"),
                             dm.StateVariable.LATERAL_POSITION, TIME)
    AX2.plot(TIME, LAT, label="Lead vehicle")
    AX2.set_xlabel("Time [s]")
    AX2.set_ylabel("Lateral position [m]")
    AX2.legend()

    plt.show()
