""" Detect all scenarios with target vehicles (in front) on the highway.

Creation date: 2020 04 28
Author(s): Erwin de Gelder

Modifications:
"""

import glob
from typing import List, NamedTuple, Tuple
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from domain_model import ActivityCategory, Actor, ActorCategory, BSplines, DetectedActivity, \
    MultiBSplines, EgoVehicle, StateVariable, Tag, VehicleType, Scenario, StaticEnvironment, \
    StaticEnvironmentCategory, Region
from activity_detector import LongitudinalStateTarget, TargetNear, LongitudinalActivity, \
    LateralActivityTarget, LateralActivityHost
from create_ngrams import FIELDNAMES_TARGET, FIELDNAMES_EGO, METADATA_TARGET, METADATA_EGO
from databaseemulator import DataBaseEmulator
from data_handler import DataHandler
from ngram import NGram
from ngram_search import find_sequence


ActivitiesResult = NamedTuple("ActivitiesResult",
                              [("actor", Actor), ("activities", List[DetectedActivity]),
                               ("acts", List[Tuple[Actor, DetectedActivity, float]])])

STAT_CATEGORY = StaticEnvironmentCategory(Region.EU_WestCentral, description="Highway",
                                          name="Highway", tags=[Tag.RoadLayout_Straight])
STAT = StaticEnvironment(STAT_CATEGORY)

TARGET = ActorCategory(VehicleType.Vehicle, name="target vehicle",
                       tags=[Tag.InitialState_LongitudinalPosition_InFrontOfEgo],)
EGO_CATEGORY = ActorCategory(VehicleType.CategoryM_PassengerCar, name="ego vehicle category")
EGO = EgoVehicle(EGO_CATEGORY, name="ego vehicle")

DEC_TARGET = ActivityCategory(MultiBSplines(2), StateVariable.LON_TARGET,
                              name="deceleration target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Braking])
ACC_TARGET = ActivityCategory(MultiBSplines(2), StateVariable.LON_TARGET,
                              name="acceleration target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Accelerating])
CRU_TARGET = ActivityCategory(MultiBSplines(2), StateVariable.LON_TARGET, name="cruising target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Cruising])
DEC_EGO = ActivityCategory(BSplines(), StateVariable.SPEED, name="deceleration ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Braking])
ACC_EGO = ActivityCategory(BSplines(), StateVariable.SPEED, name="acceleration ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Accelerating])
CRU_EGO = ActivityCategory(BSplines(), StateVariable.SPEED, name="cruising ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Cruising])
LK_TARGET = ActivityCategory(BSplines(), StateVariable.LAT_TARGET, name="lane follow target",
                             tags=[Tag.VehicleLateralActivity_GoingStraight])
LC_TARGET = ActivityCategory(BSplines(), StateVariable.LAT_TARGET, name="lane change target",
                             tags=[Tag.VehicleLateralActivity_ChangingLane])
LK_EGO = ActivityCategory(BSplines(), StateVariable.LATERAL_POSITION, name="lane keeping ego",
                          tags=[Tag.VehicleLateralActivity_GoingStraight])
LC_EGO = ActivityCategory(BSplines(), StateVariable.LATERAL_POSITION, name="lane change ego",
                          tags=[Tag.VehicleLateralActivity_ChangingLane])


def extract_near_targets(target_ngrams: NGram, ego_ngram: NGram) -> List[Tuple[int, float, float]]:
    """ Extract the scenarios.

    :param target_ngrams: The n-grams of the targets.
    :param ego_ngram: The n-gram of the ego vehicle.
    :return: A list of (target_id, start_scenario, end_scenario).
    """
    # Define the tags for the ego-braking scenario.
    target_tags = [dict(longitudinal_state=[LongitudinalStateTarget.FRONT.value],
                        near=[TargetNear.NEAR.value])]
    ego_tags = [dict(is_highway=[True])]

    # Extract the scenarios.
    scenarios = []
    for i, target in enumerate(target_ngrams.ngrams):
        search = find_sequence((target, ego_ngram.ngram), (target_tags, ego_tags))
        while search.is_found:
            scenarios.append((i, search.t_start, search.t_end))
            search = find_sequence((target, ego_ngram.ngram), (target_tags, ego_tags),
                                   t_start=search.t_end + 0.01)

    return scenarios


def fix_line_data(target: pd.DataFrame, starttime: float) -> None:
    """ A quick fix to remove any NaN from line_center of the target vehicle.

    :param target: The dataframe of the target during.
    :param starttime: The starting time of the scenario
    """
    prev_center, prev_left, prev_right = 0, 0, 0
    for i, row in enumerate(target.itertuples()):
        if i > 0:
            if np.isnan(row.line_center):
                if np.isnan(row.line_left) and not np.isnan(row.line_right):
                    target.loc[row.Index, "line_center"] = prev_center + row.line_right - prev_right
                elif np.isnan(row.line_right) and not np.isnan(row.line_left):
                    target.loc[row.Index, "line_center"] = prev_center + row.line_left - prev_left
        prev_center = target.loc[row.Index, "line_center"]
        prev_left = row.line_left
        prev_right = row.line_right

    target["line_center"].interpolate(inplace=True)
    if np.isnan(target["line_center"].iat[0]):
        target.loc[:target["line_center"].first_valid_index(), "line_center"] = \
            target.at[target["line_center"].first_valid_index(), "line_center"]

    # Check for large jumps (most likely due to ego lane changes).
    for i in target.index[target["line_center"].diff().abs() > 1].to_list():
        prev_i = target.index[target.index.get_loc(i) - 1]
        if i > starttime:
            target.loc[i:, "line_center"] += (target.at[prev_i, "line_center"] -
                                              target.at[i, "line_center"])
        else:
            target.loc[:prev_i, "line_center"] += (target.at[i, "line_center"] -
                                                   target.at[prev_i, "line_center"])


def activities_target(scenario: Tuple[int, float, float], data_handler: DataHandler,
                      target_ngrams: NGram) -> ActivitiesResult:
    """ Extract the lateral and longitudinal activities of the target vehicle.

    :param scenario: Information of the scenario: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param target_ngrams: The n-grams of the targets.
    """
    activities = []
    acts = []
    target = Actor(TARGET, properties=dict(id=scenario[0], filename=data_handler.filename),
                   name="target vehicle")

    # Store the longitudinal activities of the target.
    i_start, i_end = target_ngrams.start_and_end_indices("longitudinal_activity",
                                                         i_ngram=scenario[0],
                                                         i_start=scenario[1], i_end=scenario[2])
    for i, j in zip(i_start, i_end):
        activity_label = target_ngrams.ngrams[scenario[0]].loc[i, "longitudinal_activity"]
        activity_category = \
            (DEC_TARGET if activity_label == LongitudinalActivity.DECELERATING.value
             else ACC_TARGET if activity_label == LongitudinalActivity.ACCELERATING.value
             else CRU_TARGET)
        data = data_handler.targets[scenario[0]].loc[i:j, ["vx", "dx"]]
        n_knots = min(4, len(data) // 14)
        activity = DetectedActivity(activity_category, i, j - i,
                                    activity_category.fit(np.array(data.index), data.values,
                                                          options=dict(n_knots=n_knots)),
                                    name=activity_category.name)
        activities.append(activity)
        acts.append((target, activity, i))

    # Store the lateral activities of the target.
    fix_line_data(data_handler.targets[scenario[0]], scenario[1])
    i_start, i_end = target_ngrams.start_and_end_indices("lateral_activity",
                                                         i_ngram=scenario[0],
                                                         i_start=scenario[1], i_end=scenario[2])
    for i, j in zip(i_start, i_end):
        activity_label = target_ngrams.ngrams[scenario[0]].loc[i, "lateral_activity"]
        activity_category = \
            (LK_TARGET if activity_label == LateralActivityTarget.LANE_FOLLOWING.value
             else LC_TARGET)
        data = data_handler.targets[scenario[0]].loc[i:j, "line_center"]
        n_knots = min(4, len(data) // 14)
        activity = DetectedActivity(activity_category, i, j - i,
                                    activity_category.fit(np.array(data.index), data.values,
                                                          options=dict(n_knots=n_knots)),
                                    name=activity_category.name)
        if data_handler.targets[scenario[0]].loc[i, "lateral_activity"] in ["li", "ro"]:
            activity.name = "right lane change"
        elif data_handler.targets[scenario[0]].loc[i, "lateral_activity"] in ["ri", "lo"]:
            activity.name = "left lane change"
        activities.append(activity)
        acts.append((target, activity, i))

    return ActivitiesResult(actor=target, activities=activities, acts=acts)


def fix_ego_center(data: pd.Series, starttime):
    """ Fix the data of the distance toward the center line.

    :param data: Distance of ego vehicle toward center line.
    :param starttime: The start time of the scenario.
    """
    data.loc[data == 0] = np.nan
    data.interpolate(inplace=True)
    if np.isnan(data.iat[0]):
        data.loc[:data.first_valid_index()] = data[data.first_valid_index()]

    for i in data.index[data.diff().abs() > 1].to_list():
        prev_i = data.index[data.index.get_loc(i) - 1]
        if i > starttime:
            data.loc[i:] += data[prev_i] - data[i]
        else:
            data.loc[:prev_i] += data[i] - data[prev_i]


def activities_ego(scenario: Tuple[int, float, float], data_handler: DataHandler,
                   ego_ngram: NGram) -> ActivitiesResult:
    """ Extract the lateral and longitudinal activities of the ego vehicle.

    :param scenario: Information of the scenario: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param ego_ngram: The n-gram of the ego vehicle.
    """
    activities = []
    acts = []

    i_start, i_end = ego_ngram.start_and_end_indices("host_longitudinal_activity",
                                                     i_start=scenario[1],
                                                     i_end=scenario[2])
    for i, j in zip(i_start, i_end):
        activity_label = ego_ngram.ngram.loc[i, "host_longitudinal_activity"]
        activity_category = \
            (DEC_EGO if activity_label == LongitudinalActivity.DECELERATING.value
             else ACC_EGO if activity_label == LongitudinalActivity.ACCELERATING.value
            else CRU_EGO)
        data = data_handler.data.loc[i:j, "Host_vx"]
        n_knots = min(4, len(data) // 14)
        activity = DetectedActivity(activity_category, i, j - i,
                                    activity_category.fit(np.array(data.index), data.values,
                                                          options=dict(n_knots=n_knots)),
                                    name=activity_category.name)
        activities.append(activity)
        acts.append((EGO, activity, i))

    i_start, i_end = ego_ngram.start_and_end_indices("host_lateral_activity",
                                                     i_start=scenario[1],
                                                     i_end=scenario[2])
    data = data_handler.data.loc[i_start[0]:i_end[-1], "line_center_y"].copy()
    fix_ego_center(data, scenario[1])
    for i, j in zip(i_start, i_end):
        activity_label = ego_ngram.ngram.loc[i, "host_lateral_activity"]
        activity_category = \
            (LK_EGO if activity_label == LateralActivityHost.LANE_FOLLOWING.value
             else LC_EGO)
        subdata = data.loc[i:j]
        n_knots = min(4, len(subdata) // 14)
        activity = DetectedActivity(activity_category, i, j - i,
                                    activity_category.fit(np.array(subdata.index),
                                                          subdata.values,
                                                          options=dict(n_knots=n_knots)),
                                    name=activity_category.name)
        activities.append(activity)
        acts.append((EGO, activity, i))

    return ActivitiesResult(actor=EGO, activities=activities, acts=acts)


def process_scenario(scenario: Tuple[int, float, float], data_handler: DataHandler,
                     target_ngrams: NGram, ego_ngram: NGram, database: DataBaseEmulator) -> None:
    """ Process one cut in and add the data to the database.

    :param scenario: Information of the cut in: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param target_ngrams: The n-grams of the targets.
    :param ego_ngram: The n-gram of the ego vehicle.
    :param database: Database structure for storing the scenarios.
    """
    # Create the scenario.
    scenario_object = Scenario(scenario[1], scenario[2], STAT)

    # Obtain the activities of the target vehicle.
    target_acts = activities_target(scenario, data_handler, target_ngrams)
    database.add_item(target_acts.actor)
    scenario_object.set_actors([EGO, target_acts.actor])

    # Obtain the activities of the ego vehicle.
    ego_acts = activities_ego(scenario, data_handler, ego_ngram)

    # Write everything to the database.
    for activity in target_acts.activities+ego_acts.activities:
        database.add_item(activity)
    scenario_object.set_activities(target_acts.activities+ego_acts.activities)
    scenario_object.set_acts(target_acts.acts+ego_acts.acts)
    database.add_item(scenario_object)


def process_file(path: str, database: DataBaseEmulator):
    """ Process a single HDF5 file.

    :param path: Path of the file with the n-grams.
    :param database: Database structure for storing the scenarios.
    """
    # Extract the scenarios.
    target_ngrams = NGram(FIELDNAMES_TARGET, METADATA_TARGET)
    target_ngrams.from_hdf(path, "targets")
    ego_ngram = NGram(FIELDNAMES_EGO, METADATA_EGO)
    ego_ngram.from_hdf(path, "ego")
    scenarios = extract_near_targets(target_ngrams, ego_ngram)

    data_handler = DataHandler(os.path.join("data", "1_hdf5", os.path.basename(path)))
    for scenario in scenarios:
        if scenario[2] - scenario[1] < 0.1:
            continue  # Skip very short scenarios.
        if np.all(np.isnan(data_handler.targets[scenario[0]]["line_center"])):
            continue  # Skip scenario if no line data is available.
        try:
            process_scenario(scenario, data_handler, target_ngrams, ego_ngram, database)
        except ValueError:
            print("Error when processing scenario.")
            print("\tDatafile: {:s}".format(path))
            print("\tTarget number: {:d}".format(scenario[0]))
            print("Scenario is skipped!")
            print()


if __name__ == "__main__":
    # Instantiate the "database".
    DATABASE = DataBaseEmulator()

    # Add standard stuff.
    for item in [DEC_TARGET, ACC_TARGET, CRU_TARGET, LK_TARGET, LC_TARGET, STAT_CATEGORY, STAT,
                 TARGET, EGO_CATEGORY, EGO, DEC_EGO, ACC_EGO, CRU_EGO, LK_EGO, LC_EGO]:
        DATABASE.add_item(item)

    # Loop through the files.
    FILENAMES = glob.glob(os.path.join("data", "4_ngrams", "*.hdf5"))
    for filename in tqdm(FILENAMES):
        process_file(filename, DATABASE)

    DATABASE.to_json(os.path.join("data", "5_scenarios", "highway_target.json"), indent=4)
