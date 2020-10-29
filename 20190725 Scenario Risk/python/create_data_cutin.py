""" Create the data for Tutorial 3: Creating a *scenario* from data

Creation date: 2020 10 29
Author(s): Erwin de Gelder

Modifications:
"""

import os
import json
import numpy as np
import pandas as pd
from activity_detector import ActivityDetector, ActivityDetectorParameters
from domain_model import scenario_from_json, StateVariable


def get_noise(n, tau=0.1):
    y = np.random.randn(n)
    for i in range(1, n):
        y[i] = (1-tau)*y[i-1] + tau*y[i]
    return y


if __name__ == "__main__":
    np.random.seed(0)

    # Load the CUTIN scenario from the second tutorial.
    FILENAME_CUTIN = os.path.join("..", "20200729 Domain Model", "examples",
                                  "cutin_quantitative.json")
    with open(FILENAME_CUTIN, "r") as FILE:
        CUTIN = scenario_from_json(json.load(FILE))
    EGO_VEHICLE = CUTIN.get_actor_by_name("Ego vehicle")
    TARGET_VEHICLE = CUTIN.get_actor_by_name("Target vehicle")

    time = np.arange(CUTIN.get_tstart(), CUTIN.get_tend()+0.01, 0.01)
    vego = CUTIN.get_state(EGO_VEHICLE, StateVariable.SPEED, time)
    yego = CUTIN.get_state(EGO_VEHICLE, StateVariable.LATERAL_POSITION, time)
    vtarget = CUTIN.get_state(TARGET_VEHICLE, StateVariable.SPEED, time)
    ytarget = CUTIN.get_state(TARGET_VEHICLE, StateVariable.LATERAL_POSITION, time)
    df = pd.DataFrame(vego+get_noise(len(vego))/4+get_noise(len(vego), tau=0.5)/10,
                      columns=["v_ego"], index=time)
    df["Target_1_id"] = 0
    df["Target_1_vx"] = 0
    df["Target_0_id"] = 1
    df["Target_0_vx"] = (vtarget + get_noise(len(vego), tau=0.02)*2 +
                         get_noise(len(vego), tau=0.5)/20)
    df["lines_0_c0"] = 3 - (yego + get_noise(len(time), tau=0.02)*.2 +
                            get_noise(len(vego), tau=.5)/10)
    df["lines_1_c0"] = -(yego + get_noise(len(time), tau=0.02)*.2 + get_noise(len(vego), tau=.5)/10)
    df["lines_0_c1"], df["lines_1_c1"], df["lines_0_c2"], df["lines_1_c2"] = 0, 0, 0, 0
    df["lines_0_c3"], df["lines_1_c3"] = 0, 0
    df["lines_0_quality"] = 3
    df["lines_1_quality"] = 3
    df["Target_1_dy"] = 0
    df["Target_0_dy"] = (ytarget - yego + get_noise(len(time), tau=0.03) +
                         get_noise(len(time), tau=.5)/10)
    df["Target_1_dx"] = 0
    df["Target_0_dx"] = (TARGET_VEHICLE.initial_states[0].value +
                         (np.cumsum(vtarget)-np.cumsum(vego))*0.01 +
                         get_noise(len(time), tau=0.03)*2 + get_noise(len(vego), tau=.5)/8)
    a = ActivityDetector(df, ActivityDetectorParameters(host_lon_vel="v_ego"))

    a.lon_activities_host(plot=True)
    a.set_lon_activities_host()
    a.lon_activities_target_i(0, plot=True)
    a.set_lat_activities_host()
    a.lat_activities_target_i(i=0, plot=True)
    a.set_target_activities(i=0)

    new = pd.DataFrame(df[["v_ego", "lines_0_c0", "lines_1_c0", "Target_0_dx", "Target_0_vx",
                           "host_longitudinal_activity", "host_lateral_activity"]].values,
                       index=df.index,
                       columns=["v_ego", "line_left", "line_right", "d_target", "v_target",
                                "act_lon_ego", "act_lat_ego"])
    new["line_left_target"] = a.targets[0]["line_left"]
    new["line_right_target"] = a.targets[0]["line_right"]
    new["act_lon_target"] = a.targets[0]["longitudinal_activity"]
    new["act_lat_target"] = a.targets[0]["lateral_activity"]
    new.to_csv(os.path.join("examples", "data_cutin_scenario.csv"))

