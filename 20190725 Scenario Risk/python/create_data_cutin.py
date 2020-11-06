""" Create the data for Tutorial 3: Creating a *scenario* from data

Creation date: 2020 10 29
Author(s): Erwin de Gelder

Modifications:
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from activity_detector import ActivityDetector, ActivityDetectorParameters
from domain_model import scenario_from_json, StateVariable


def get_noise(nsamples, tau=0.1):
    """ Create noise data.

    :param nsamples: Number of samples.
    :param tau: Time constant that determines correlation between consecutive
        samples.
    :return: Noise data.
    """
    output = np.random.randn(nsamples)
    output[0] = tau*output[0]
    for i in range(1, nsamples):
        output[i] = (1-tau)*output[i-1] + tau*output[i]
    return output


if __name__ == "__main__":
    np.random.seed(0)

    # Load the CUTIN scenario from the second tutorial.
    FILENAME_CUTIN = os.path.join("..", "..", "20200729 Domain Model", "examples",
                                  "cutin_quantitative.json")
    with open(FILENAME_CUTIN, "r") as FILE:
        CUTIN = scenario_from_json(json.load(FILE))
    EGO_VEHICLE = CUTIN.get_actor_by_name("ego vehicle")
    TARGET_VEHICLE = CUTIN.get_actor_by_name("target vehicle")

    TIME = np.arange(CUTIN.get_tstart(), CUTIN.get_tend()+0.01, 0.01)
    VEGO = CUTIN.get_state(EGO_VEHICLE, StateVariable.SPEED, TIME)
    YEGO = CUTIN.get_state(EGO_VEHICLE, StateVariable.LATERAL_POSITION, TIME)
    VTARGET = CUTIN.get_state(TARGET_VEHICLE, StateVariable.SPEED, TIME)
    YTARGET = CUTIN.get_state(TARGET_VEHICLE, StateVariable.LATERAL_POSITION, TIME)
    DATA = pd.DataFrame(VEGO+get_noise(len(VEGO))/4+get_noise(len(VEGO), tau=0.5)/10,
                        columns=["v_ego"], index=TIME)
    DATA["Target_1_id"] = 0
    DATA["Target_1_vx"] = 0
    DATA["Target_0_id"] = 1
    DATA["Target_0_vx"] = (VTARGET+get_noise(len(VEGO), tau=0.02)*2+
                           get_noise(len(VEGO), tau=0.5)/20)
    DATA["lines_0_c0"] = 3-(YEGO+get_noise(len(TIME), tau=0.02)*.2+
                            get_noise(len(VEGO), tau=.5)/10)
    DATA["lines_1_c0"] = -(YEGO+get_noise(len(TIME), tau=0.02)*.2+get_noise(len(VEGO), tau=.5)/10)
    DATA["lines_0_c1"], DATA["lines_1_c1"], DATA["lines_0_c2"], DATA["lines_1_c2"] = 0, 0, 0, 0
    DATA["lines_0_c3"], DATA["lines_1_c3"] = 0, 0
    DATA["lines_0_quality"] = 3
    DATA["lines_1_quality"] = 3
    DATA["Target_1_dy"] = 0
    DATA["Target_0_dy"] = (YTARGET-YEGO+get_noise(len(TIME), tau=0.03)+
                           get_noise(len(TIME), tau=.5)/10)
    DATA["Target_1_dx"] = 0
    DATA["Target_0_dx"] = (TARGET_VEHICLE.initial_states[0].value+
                           (np.cumsum(VTARGET)-np.cumsum(VEGO))*0.01+
                           get_noise(len(TIME), tau=0.03)*2+get_noise(len(VEGO), tau=.5)/8)
    AD = ActivityDetector(DATA, ActivityDetectorParameters(host_lon_vel="v_ego"))

    plt.figure()
    AD.lon_activities_host(plot=True)
    AD.set_lon_activities_host()
    plt.figure()
    AD.lon_activities_target_i(0, plot=True)
    AD.set_lat_activities_host()
    plt.figure()
    AD.lat_activities_target_i(i=0, plot=True)
    AD.set_target_activities(i=0)

    NEW_DATA = pd.DataFrame(DATA[["v_ego", "lines_0_c0", "lines_1_c0", "Target_0_dx", "Target_0_vx",
                                  "host_longitudinal_activity", "host_lateral_activity"]].values,
                            index=DATA.index,
                            columns=["v_ego", "line_left", "line_right", "d_target",
                                     "v_target", "act_lon_ego", "act_lat_ego"])
    NEW_DATA["line_left_target"] = AD.targets[0]["line_left"]
    NEW_DATA["line_right_target"] = AD.targets[0]["line_right"]
    NEW_DATA["act_lon_target"] = AD.targets[0]["longitudinal_activity"]
    NEW_DATA["act_lat_target"] = AD.targets[0]["lateral_activity"]
    print(NEW_DATA.head())
    print(NEW_DATA.tail())
    NEW_DATA.to_csv(os.path.join("..", "..", "20200729 Domain Model", "examples",
                                 "data_cutin_scenario.csv"))

    plt.show()
