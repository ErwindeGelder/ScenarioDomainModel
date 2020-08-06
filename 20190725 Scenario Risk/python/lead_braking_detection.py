""" Detect all lead braking activities and store them in a kind-of database.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
"""

import glob
import os
from typing import List, NamedTuple, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from domain_model import BSplines, ActivityCategory, StateVariable, Tag, \
    DetectedActivity, StaticEnvironmentCategory, StaticEnvironment, Region, ActorCategory, Actor, \
    VehicleType, EgoVehicle, Scenario, MultiBSplines
from activity_detector import LeadVehicle, LongitudinalActivity, LateralActivityTarget
from create_ngrams import FIELDNAMES_EGO, METADATA_EGO, FIELDNAMES_TARGET, METADATA_TARGET
from databaseemulator import DataBaseEmulator
from data_handler import DataHandler
from ngram import NGram
from ngram_search import find_sequence


ActivitiesResult = NamedTuple("ActivitiesResult",
                              [("actor", Actor), ("activities", List[DetectedActivity]),
                               ("acts", List[Tuple[Actor, DetectedActivity, float]])])


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
LK_TARGET = ActivityCategory(BSplines(), StateVariable.LATERAL_POSITION, name="lane keeping target",
                             tags=[Tag.VehicleLateralActivity_GoingStraight])
LC_TARGET = ActivityCategory(BSplines(), StateVariable.LAT_TARGET, name="lane change target",
                             tags=[Tag.VehicleLateralActivity_ChangingLane])
LK_EGO = ActivityCategory(BSplines(), StateVariable.LATERAL_POSITION, name="lane keeping ego",
                          tags=[Tag.VehicleLateralActivity_GoingStraight])
LC_EGO = ActivityCategory(BSplines(), StateVariable.LATERAL_POSITION, name="lane change ego",
                          tags=[Tag.VehicleLateralActivity_ChangingLane])
STAT_CATEGORY = StaticEnvironmentCategory(Region.EU_WestCentral, description="Highway",
                                          name="Highway", tags=[Tag.RoadLayout_Straight])
STAT = StaticEnvironment(STAT_CATEGORY)
EGO_CATEGORY = ActorCategory(VehicleType.CategoryM_PassengerCar, name="ego vehicle category")
EGO = EgoVehicle(EGO_CATEGORY, name="ego vehicle")
TARGET = ActorCategory(VehicleType.Vehicle, name="lead vehicle",
                       tags=[Tag.LeadVehicle_Following,
                             Tag.InitialState_Direction_SameAsEgo],)


def process_file(path: str, database: DataBaseEmulator):
    """ Process a single HDF5 file.

    :param path: Path of the file with the n-grams.
    :param database: Database structure for storing the scenarios.
    """
    # Extract the deceleration activities of the ego vehicle.
    target_ngrams = NGram(FIELDNAMES_TARGET, METADATA_TARGET)
    target_ngrams.from_hdf(path, "targets")
    ego_ngram = NGram(FIELDNAMES_EGO, METADATA_EGO)
    ego_ngram.from_hdf(path, "ego")
    brakings = extract_lead_braking(target_ngrams)

    # Store each scenario.
    data_handler = DataHandler(os.path.join("data", "1_hdf5", os.path.basename(path)))
    skipped = 0
    for braking in brakings:
        if braking[2] - braking[1] < 0.1:
            skipped += 1
            continue  # Skip very short scenarios.
        process_scenario(braking, data_handler, target_ngrams, ego_ngram, database)
    return len(brakings) - skipped


def extract_lead_braking(target_ngrams: NGram) -> List[Tuple[int, float, float]]:
    """ Extract cut ins.

    :param target_ngrams: The n-gram of the target vehicles.
    :return: A list of (itarget, start_braking, end_braking).
    """
    # Define the tags for the ego-braking scenario.
    target_tags = [dict(lead_vehicle=[LeadVehicle.LEAD.value],
                        longitudinal_activity=[LongitudinalActivity.DECELERATING.value])]

    # Extract the scenarios where the target vehicle is decelerating.
    scenarios = []
    for i, target in enumerate(target_ngrams.ngrams):
        search = find_sequence((target,), (target_tags,))
        while search.is_found:
            scenarios.append((i, search.t_start, search.t_end))
            search = find_sequence((target,), (target_tags,), t_start=search.t_end+0.1)

    return scenarios


def process_scenario(scenario: Tuple[int, float, float], data_handler: DataHandler,
                     target_ngrams: NGram, ego_ngram: NGram, database: DataBaseEmulator) -> None:
    """ Process one scenario and add the data to the database.

    :param scenario: Information of the scenario: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param target_ngrams: The n-grams of the targets.
    :param ego_ngram: The n-gram of the ego vehicle.
    :param database: Database structure for storing the scenarios.
    """
    # Create the scenario.
    scenario_object = Scenario(scenario[1], scenario[2], STAT)

    # Store the longitudinal activities of the target vehicle.
    target_acts = activities_target(scenario, data_handler, target_ngrams)
    database.add_item(target_acts.actor)
    scenario_object.set_actors([EGO, target_acts.actor])

    # Store the lateral and longitudinal activities of the ego vehicle.
    ego_acts = activities_ego(scenario, data_handler, ego_ngram)

    # Write everything to the database.
    for activity in target_acts.activities+ego_acts.activities:
        database.add_item(activity)
    scenario_object.set_activities(target_acts.activities+ego_acts.activities)
    scenario_object.set_acts(target_acts.acts+ego_acts.acts)
    database.add_item(scenario_object)


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
    for type_of_activity in ["lateral_activity", "longitudinal_activity"]:
        i_start, i_end = target_ngrams.start_and_end_indices(type_of_activity, i_ngram=scenario[0],
                                                             i_start=scenario[1], i_end=scenario[2])
        for i, j in zip(i_start, i_end):
            activity_label = target_ngrams.ngrams[scenario[0]].loc[i, type_of_activity]
            if type_of_activity == "longitudinal_activity":
                activity_category = \
                    (DEC_TARGET if activity_label == LongitudinalActivity.DECELERATING.value
                     else ACC_TARGET if activity_label == LongitudinalActivity.ACCELERATING.value
                     else CRU_TARGET)
                data = data_handler.targets[scenario[0]].loc[i:j, ["vx", "dx"]]
            else:
                activity_category = \
                    (LK_TARGET if activity_label == LateralActivityTarget.LANE_FOLLOWING.value
                     else LC_TARGET)
                data = data_handler.targets[scenario[0]].loc[scenario[1]:scenario[2]]
                fix_line_data(data_handler.targets[scenario[0]], scenario[1], scenario[2])
                data = data["line_center"]

            if np.any(np.isnan(data)):
                raise ValueError("NaN data!!!")
            n_knots = min(4, len(data)//10)
            activity = DetectedActivity(activity_category, i, j-i,
                                        activity_category.fit(np.array(data.index), data.values,
                                                              options=dict(n_knots=n_knots)),
                                        name=activity_category.name)
            activities.append(activity)
            acts.append((target, activity, i))
    return ActivitiesResult(actor=target, activities=activities, acts=acts)


def fix_line_data(target: pd.DataFrame, i: float, j: float) -> None:
    """ A quick fix to remove any NaN from line_center of the target vehicle.

    :param target: The dataframe of the target during.
    :param i: Starting index.
    :param j: End index.
    """
    lanewidth = target.loc[i:j, "line_left"] - target.loc[i:j, "line_right"]
    lanewidth = np.mean(lanewidth[np.logical_not(np.isnan(lanewidth))])
    if target["lateral_activity"].iloc[0] == "ri":
        target.loc[i:j, "line_center"] = target.loc[i:j, "line_right"] + lanewidth/2
    else:
        target.loc[i:j, "line_center"] = target.loc[i:j, "line_left"] - lanewidth/2
    target["line_center"].interpolate(inplace=True)


def activities_ego(scenario: Tuple[int, float, float], data_handler: DataHandler,
                   ego_ngram: NGram) -> ActivitiesResult:
    """ Extract the lateral and longitudinal activities of the ego vehicle.

    :param scenario: Information of the scenario: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param ego_ngram: The n-gram of the ego vehicle.
    """
    activities = []
    acts = []
    for type_of_activity in ["host_lateral_activity", "host_longitudinal_activity"]:
        i_start, i_end = ego_ngram.start_and_end_indices(type_of_activity, i_start=scenario[1],
                                                         i_end=scenario[2])
        for i, j in zip(i_start, i_end):
            activity_label = ego_ngram.ngram.loc[i, type_of_activity]
            if type_of_activity == "host_longitudinal_activity":
                activity_category = \
                    (DEC_EGO if activity_label == LongitudinalActivity.DECELERATING.value
                     else ACC_EGO if activity_label == LongitudinalActivity.ACCELERATING.value
                     else CRU_EGO)
                data = data_handler.data.loc[i:j, "Host_vx"]
            else:
                activity_category = LK_EGO
                data = data_handler.data.loc[i:j, "line_center_y"]

            if np.any(np.isnan(data)):
                raise ValueError("NaN data!!!")

            n_knots = min(4, len(data)//10)
            activity = DetectedActivity(activity_category, i, j - i,
                                        activity_category.fit(np.array(data.index), data.values,
                                                              options=dict(n_knots=n_knots)),
                                        name=activity_category.name)
            activities.append(activity)
            acts.append((EGO, activity, i))
    return ActivitiesResult(actor=EGO, activities=activities, acts=acts)


if __name__ == "__main__":
    # Instantiate the "database".
    DATABASE = DataBaseEmulator()

    # Add standard stuff.
    for item in [STAT_CATEGORY, STAT, EGO_CATEGORY, EGO, TARGET,
                 DEC_TARGET, CRU_TARGET, ACC_TARGET, LK_TARGET, LC_TARGET,
                 DEC_EGO, CRU_EGO, ACC_EGO, LK_EGO, LC_EGO]:
        DATABASE.add_item(item)

    # Loop through the files.
    FILENAMES = glob.glob(os.path.join("data", "4_ngrams", "*.hdf5"))
    N_SCENARIOS = 0
    for filename in tqdm(FILENAMES):
        N_SCENARIOS += process_file(filename, DATABASE)
    print("Number of scenarios: {:d}".format(N_SCENARIOS))

    FOLDER = os.path.join("data", "5_scenarios")
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    DATABASE.to_json(os.path.join(FOLDER, "lead_braking.json"), indent=4)
