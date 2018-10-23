from ActivityCategory import ActivityCategory
from ActorCategory import ActorCategory
from Model import Model
from Activity import DetectedActivity
from Actor import Actor
from Scenario import Scenario
import matplotlib.pyplot as plt
import os
import json

# Check if folder exists to store images
figfolder = 'examples'
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# Define the qualitative activities
accelerating = ActivityCategory("accelerating", Model("Spline3Knots"), "x",
                                tags=["Long. activity - Driving forward - Accelerating"])
braking = ActivityCategory("braking", Model("Spline3Knots"), "x",
                           tags=["Long. activity - Driving forward - Braking"])
cruising = ActivityCategory("cruising", Model("Linear"), "x",
                            tags=["Long. activity - Driving forward - Cruising"])

# Define activities
ego_acceleration1 = DetectedActivity("ego_accelerating", accelerating, 0, 8.97,
                                     {"xstart": 0, "xend": 133, "a1": -7.55e-2, "b1": 3.62e-1, "c1": 7.07e-1, "d1": 0,
                                      "a2": -2.18e-2, "b2": 2.82e-1, "c2": 7.47e-1, "d2": -6.73e-3})
ego_braking = DetectedActivity("ego_braking", braking, 9.00, 1.98,
                               {"xstart": 133, "xend": 168, "a1": 1.56e-2, "b1": -6.27e-2, "c1": 1.04, "d1": 0,
                                "a2": 3.31e-2, "b2": -8.89e-2, "c2": 1.06, "d2": -2.18e-3})
ego_cruising1 = DetectedActivity("ego_cruising", cruising, 11, 6.98, {"xstart": 168, "xend": 287})
ego_acceleration2 = DetectedActivity("ego_accelerating2", accelerating, 18, 4.98,
                                     {"xstart": 288, "xend": 381, "a1": -6.8e-2, "b1": 1.45e-1, "c1": 9.17e-1, "d1": 0,
                                      "a2": -3.12e-2, "b2": 9.09e-2, "c2": 9.45e-1, "d2": -4.6e-3})
ego_cruising2 = DetectedActivity("ego_cruising2", cruising, 23, 19.3, {"xstart": 381, "xend": 756})

# Plot the longitudinal position of the ego vehicle over time
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for activity in [ego_acceleration1, ego_braking, ego_cruising1, ego_acceleration2, ego_cruising2]:
    activity.plot_state(ax=ax1)
    activity.plot_state_dot(ax=ax2)
ax1.set_xlabel('Time [s]')
ax2.set_xlabel('Time [s]')
ax1.set_ylabel('Position [m]')
ax2.set_ylabel('Speed [m/s]')
f.savefig(os.path.join(figfolder, 'ego_activities'))

# Plot the state of an actor
qualitative_ego = ActorCategory("sedan", "Passenger car (M1)", tags=["Passenger car (M1)"])
ego = Actor("ego", qualitative_ego, tags=["Ego"])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

print("Tags of ego vehicle:")
for tag in ego.get_tags():
    print(" - {:s}".format(tag))

qualitative_pickup = ActorCategory("pickup", "Truck", tags=["Truck"])
pickup_cruising = DetectedActivity("pickup_cruising", cruising, 0.8, 32.6, {"xstart": 117, "xend": 556})
pickup = Actor("pickup", qualitative_pickup, tags=["Direction - Same as ego", "Lat. pos. - Same lane",
                                                   "Long. pos. - In front of ego", "Appearing - Gap-closing",
                                                   "Lat. act. - Straight", "Lat. act. - Going forward"])
print("Tags of pickup truck:")
for tag in pickup.get_tags():
    print(" - {:s}".format(tag))

# Define the scenario
activities = [ego_acceleration1, ego_braking, ego_cruising1, ego_acceleration2, ego_cruising2]
acts = [[ego, activity, activity.tstart] for activity in activities]
scenario = Scenario("example", 0, ego_cruising2.tend, actors=[ego], activities=activities, acts=acts)
with open(os.path.join("examples", "example.json"), "w") as f:
    print(scenario.to_json())
    json.dump(scenario.to_json(), f)
