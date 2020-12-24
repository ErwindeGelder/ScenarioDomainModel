""" Detect all lead braking activities and store them in a kind-of database.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
2020 08 12 Update functions a bit. Same result, but now it can also be used for other scenarios.
2020 12 01 Update based on updated domain model.
"""

import glob
import os
from typing import Callable, List, NamedTuple, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from domain_model import Splines, ActivityCategory, StateVariable, Tag, \
    Activity, PhysicalElementCategory, PhysicalElement, ActorCategory, Actor, \
    VehicleType, EgoVehicle, Scenario, MultiBSplines, DocumentManagement
from activity_detector import LeadVehicle, LongitudinalActivity, LateralActivityTarget
from create_ngrams import FIELDNAMES_EGO, METADATA_EGO, FIELDNAMES_TARGET, METADATA_TARGET
from data_handler import DataHandler
from ngram import NGram
from ngram_search import find_sequence


ActivitiesResult = NamedTuple("ActivitiesResult",
                              [("actor", Actor), ("activities", List[Activity]),
                               ("acts", List[Tuple[Actor, Activity]])])


MULTISPLINES = MultiBSplines(2)
DEC_TARGET = ActivityCategory(MULTISPLINES, StateVariable.LON_TARGET,
                              name="deceleration target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Braking])
ACC_TARGET = ActivityCategory(MULTISPLINES, StateVariable.LON_TARGET,
                              name="acceleration target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Accelerating])
CRU_TARGET = ActivityCategory(MULTISPLINES, StateVariable.LON_TARGET, name="cruising target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Cruising])
SPLINES = Splines()
DEC_EGO = ActivityCategory(SPLINES, StateVariable.SPEED, name="deceleration ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Braking])
ACC_EGO = ActivityCategory(SPLINES, StateVariable.SPEED, name="acceleration ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Accelerating])
CRU_EGO = ActivityCategory(SPLINES, StateVariable.SPEED, name="cruising ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Cruising])
LK_TARGET = ActivityCategory(SPLINES, StateVariable.LATERAL_POSITION, name="lane keeping target",
                             tags=[Tag.VehicleLateralActivity_GoingStraight])
LC_TARGET = ActivityCategory(SPLINES, StateVariable.LAT_TARGET, name="lane change target",
                             tags=[Tag.VehicleLateralActivity_ChangingLane])
LK_EGO = ActivityCategory(SPLINES, StateVariable.LATERAL_POSITION, name="lane keeping ego",
                          tags=[Tag.VehicleLateralActivity_GoingStraight])
LC_EGO = ActivityCategory(SPLINES, StateVariable.LATERAL_POSITION, name="lane change ego",
                          tags=[Tag.VehicleLateralActivity_ChangingLane])
STAT_CATEGORY = PhysicalElementCategory(description="Highway", name="Highway",
                                        tags=[Tag.RoadLayout_Straight])
STAT = PhysicalElement(STAT_CATEGORY)
EGO_CATEGORY = ActorCategory(VehicleType.CategoryM_PassengerCar, name="ego vehicle category")
EGO = EgoVehicle(EGO_CATEGORY, name="ego vehicle")
TARGET = ActorCategory(VehicleType.Vehicle, name="lead vehicle",
                       tags=[Tag.LeadVehicle_Following,
                             Tag.InitialState_Direction_SameAsEgo],)


def process_file(path: str, database: DocumentManagement, func_scen_extraction: Callable):
    """ Process a single HDF5 file.

    :param path: Path of the file with the n-grams.
    :param database: Database structure for storing the scenarios.
    :param func_scen_extraction: Function that is used to extract the scenarios from the n-grams.
    """
    # Extract the deceleration activities of the ego vehicle.
    target_ngrams = NGram(FIELDNAMES_TARGET, METADATA_TARGET)
    target_ngrams.from_hdf(path, "targets")
    ego_ngram = NGram(FIELDNAMES_EGO, METADATA_EGO)
    ego_ngram.from_hdf(path, "ego")
    brakings = func_scen_extraction(ego_ngram, target_ngrams)

    # Store each scenario.
    data_handler = DataHandler(os.path.join("data", "1_hdf5", os.path.basename(path)))
    for braking in brakings:
        process_scenario(braking, data_handler, target_ngrams, ego_ngram, database)
    return len(brakings)


def extract_lead_braking(ego_ngram: NGram, target_ngrams: NGram) -> List[Tuple[int, float, float]]:
    """ Extract cut ins.

    :param ego_ngram: The n-gram of the ego vehicle (not used).
    :param target_ngrams: The n-gram of the target vehicles.
    :return: A list of (itarget, start_braking, end_braking).
    """
    _ = ego_ngram
    # Define the tags for the ego-braking scenario.
    target_tags = [dict(lead_vehicle=[LeadVehicle.LEAD.value],
                        longitudinal_activity=[LongitudinalActivity.DECELERATING.value])]

    # Extract the scenarios where the target vehicle is decelerating.
    scenarios = []
    for i, target in enumerate(target_ngrams.ngrams):
        search = find_sequence((target,), (target_tags,))
        while search.is_found:
            if search.t_end - search.t_start > 0.1:  # Skip very short scenarios.
                scenarios.append((i, search.t_start, search.t_end))
            search = find_sequence((target,), (target_tags,), t_start=search.t_end+0.1)

    return scenarios


def process_scenario(scenario: Tuple[int, float, float], data_handler: DataHandler,
                     target_ngrams: NGram, ego_ngram: NGram, database: DocumentManagement) -> None:
    """ Process one scenario and add the data to the database.

    :param scenario: Information of the scenario: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param target_ngrams: The n-grams of the targets.
    :param ego_ngram: The n-gram of the ego vehicle.
    :param database: Database structure for storing the scenarios.
    """
    # Create the scenario.
    scenario_object = Scenario(start=scenario[1], end=scenario[2], physical_elements=[STAT])

    # Store the longitudinal activities of the target vehicle.
    target_acts = activities_target(scenario, data_handler, target_ngrams)
    database.add_item(target_acts.actor, include_attributes=True)
    scenario_object.set_actors([EGO, target_acts.actor])

    # Store the lateral and longitudinal activities of the ego vehicle.
    ego_acts = activities_ego(scenario, data_handler, ego_ngram)

    # Write everything to the database.
    for activity in target_acts.activities+ego_acts.activities:
        database.add_item(activity, include_attributes=True)
    scenario_object.set_activities(target_acts.activities+ego_acts.activities)
    scenario_object.set_acts(target_acts.acts+ego_acts.acts)
    database.add_item(scenario_object, include_attributes=True)


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
        start = i_start[0]
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
            activity = Activity(activity_category, start=start, end=j,
                                parameters=activity_category.fit(np.array(data.index), data.values,
                                                                 n_knots=n_knots),
                                name=activity_category.name)
            start = activity.end
            activities.append(activity)
            acts.append((target, activity))
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
        start = i_start[0]
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
            activity = Activity(activity_category, start=start, end=j,
                                parameters=activity_category.fit(np.array(data.index), data.values,
                                                                 n_knots=n_knots),
                                name=activity_category.name)
            start = activity.end
            activities.append(activity)
            acts.append((EGO, activity))
    return ActivitiesResult(actor=EGO, activities=activities, acts=acts)


if __name__ == "__main__":
    # Instantiate the "database".
    DATABASE = DocumentManagement()

    # Add standard stuff.
    for item in [MULTISPLINES, SPLINES, STAT_CATEGORY, STAT, EGO_CATEGORY, EGO, TARGET,
                 DEC_TARGET, CRU_TARGET, ACC_TARGET, LK_TARGET, LC_TARGET,
                 DEC_EGO, CRU_EGO, ACC_EGO, LK_EGO, LC_EGO]:
        DATABASE.add_item(item, include_attributes=True)

    # Loop through the files.
    FILENAMES = glob.glob(os.path.join("data", "4_ngrams", "*.hdf5"))
    N_SCENARIOS = 0
    for filename in tqdm(FILENAMES):
        N_SCENARIOS += process_file(filename, DATABASE, extract_lead_braking)
    print("Number of scenarios: {:d}".format(N_SCENARIOS))

    FOLDER = os.path.join("data", "5_scenarios")
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    DATABASE.to_json(os.path.join(FOLDER, "lead_braking2.json"), indent=4)
