from QualitativeActivity import QualitativeActivity
from QualitativeActor import QualitativeActor
from Model import Model
from Activity import DetectedActivity
from Actor import Actor
import matplotlib.pyplot as plt
import os

# Check if folder exists to store images
figfolder = 'examples'
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# Define the qualitative activities
accelerating = QualitativeActivity("accelerating", Model("Spline3Knots"), "x",
                                   tags=["Long. activity - Driving forward - Accelerating"])
braking = QualitativeActivity("braking", Model("Spline3Knots"), "x",
                              tags=["Long. activity - Driving forward - Braking"])
cruising = QualitativeActivity("cruising", Model("Linear"), "x",
                               tags=["Long. activity - Driving forward - Cruising"])

# Define activities
ego_acceleration1 = DetectedActivity(accelerating, 0, 8.97,
                                     {"xstart": 0, "xend": 133, "a1": -7.55e-2, "b1": 3.62e-1, "c1": 7.07e-1, "d1": 0,
                                      "a2": -2.18e-2, "b2": 2.82e-1, "c2": 7.47e-1, "d2": -6.73e-3})
ego_braking = DetectedActivity(braking, 9.00, 1.98,
                               {"xstart": 133, "xend": 168, "a1": 1.56e-2, "b1": -6.27e-2, "c1": 1.04, "d1": 0,
                                "a2": 3.31e-2, "b2": -8.89e-2, "c2": 1.06, "d2": -2.18e-3})
ego_cruising1 = DetectedActivity(cruising, 11, 6.98, {"xstart": 168, "xend": 287})
ego_acceleration2 = DetectedActivity(accelerating, 18, 4.98,
                                     {"xstart": 288, "xend": 381, "a1": -6.8e-2, "b1": 1.45e-1, "c1": 9.17e-1, "d1": 0,
                                      "a2": -3.12e-2, "b2": 9.09e-2, "c2": 9.45e-1, "d2": -4.6e-3})
ego_cruising2 = DetectedActivity(cruising, 23, 19.3, {"xstart": 381, "xend": 756})

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
qualitative_ego = QualitativeActor("sedan", "Passenger car (M1)", tags=["Passenger car (M1)"])
ego = Actor("ego", qualitative_ego, tags=["Ego"])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

print("Tags of ego vehicle:")
for tag in ego.get_tags():
    print(" - {:s}".format(tag))
