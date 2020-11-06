""" Detect all cut ins and store them in a kind-of database.

Creation date: 2020 03 23
Author(s): Erwin de Gelder

Modifications:
2020 03 27: Enable the instantiation of objects after reading in a database.
2020 03 29: Use a variable number of knots to prevent error for very short activities.
2020 04 04: Various bug fixes.
2020 11 06: Work with updated domain model.
"""

import glob
import os
from typing import List, NamedTuple, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from domain_model import MultiBSplines, Splines, ActivityCategory, StateVariable, Tag, \
    Activity, PhysicalElementCategory, PhysicalElement, ActorCategory, Actor, \
    VehicleType, EgoVehicle, Scenario, DocumentManagement
from activity_detector import LateralActivityTarget, LateralActivityHost, LeadVehicle, \
    LongitudinalActivity
from create_ngrams import FIELDNAMES_TARGET, FIELDNAMES_EGO, METADATA_TARGET, METADATA_EGO
from data_handler import DataHandler
from ngram import NGram
from ngram_search import find_sequence


ActivitiesResult = NamedTuple("ActivitiesResult", [("actor", Actor),
                                                   ("activities", List[Activity]),
                                                   ("acts", List[Tuple[Actor, Activity]])])

MULTIBSPLINES = MultiBSplines(2)
DEC_TARGET = ActivityCategory(MULTIBSPLINES, StateVariable.LON_TARGET,
                              name="deceleration target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Braking])
ACC_TARGET = ActivityCategory(MULTIBSPLINES, StateVariable.LON_TARGET,
                              name="acceleration target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Accelerating])
CRU_TARGET = ActivityCategory(MULTIBSPLINES, StateVariable.LON_TARGET, name="cruising target",
                              tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Cruising])
SPLINES = Splines()
DEC_EGO = ActivityCategory(SPLINES, StateVariable.SPEED, name="deceleration ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Braking])
ACC_EGO = ActivityCategory(SPLINES, StateVariable.SPEED, name="acceleration ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Accelerating])
CRU_EGO = ActivityCategory(SPLINES, StateVariable.SPEED, name="cruising ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Cruising])
LC_TARGET = ActivityCategory(SPLINES, StateVariable.LAT_TARGET, name="lane change target",
                             tags=[Tag.VehicleLateralActivity_ChangingLane])
LK_EGO = ActivityCategory(SPLINES, StateVariable.LATERAL_POSITION, name="lane keeping ego",
                          tags=[Tag.VehicleLateralActivity_GoingStraight])
STAT_CATEGORY = PhysicalElementCategory(description="Highway", name="Highway",
                                        tags=[Tag.RoadLayout_Straight])
STAT = PhysicalElement(STAT_CATEGORY)
TARGET = ActorCategory(VehicleType.Vehicle, name="cut-in vehicle",
                       tags=[Tag.LeadVehicle_Appearing_CuttingIn,
                             Tag.InitialState_Direction_SameAsEgo],)
EGO_CATEGORY = ActorCategory(VehicleType.CategoryM_PassengerCar, name="ego vehicle category")
EGO = EgoVehicle(EGO_CATEGORY, name="ego vehicle")


def extract_cutins(target_ngrams: NGram, ego_ngram: NGram) -> List[Tuple[int, float, float]]:
    """ Extract cut ins.

    :param target_ngrams: The n-grams of the targets.
    :param ego_ngram: The n-gram of the ego vehicle.
    :return: A list of (target_id, start_cutin, end_cutin).
    """
    # Define the tags for the cut-in scenario.
    target_tags = [dict(lateral_activity=[LateralActivityTarget.LEFT_CUT_IN.value,
                                          LateralActivityTarget.RIGHT_CUT_IN.value],
                        lead_vehicle=[LeadVehicle.NOLEAD.value]),
                   dict(lateral_activity=[LateralActivityTarget.LEFT_CUT_IN.value,
                                          LateralActivityTarget.RIGHT_CUT_IN.value],
                        lead_vehicle=[LeadVehicle.LEAD.value])]
    ego_tags = [dict(host_lateral_activity=[LateralActivityHost.LANE_FOLLOWING.value],
                     is_highway=[True]),
                dict(host_lateral_activity=[LateralActivityHost.LANE_FOLLOWING.value],
                     is_highway=[True])]

    # Extract the cut ins.
    cutins = []
    for i, target in enumerate(target_ngrams.ngrams):
        search = find_sequence((target, ego_ngram.ngram), (target_tags, ego_tags))
        while search.is_found:
            cutins.append((i, search.t_start, search.t_end))
            search = find_sequence((target, ego_ngram.ngram), (target_tags, ego_tags),
                                   t_start=search.t_end + 5)
    return cutins


def fix_line_data(target: pd.DataFrame, activity: str, i: float, j: float) -> None:
    """ A quick fix to remove any NaN from line_center of the target vehicle.

    :param target: The dataframe of the target during.
    :param activity: 'ri' or 'li'.
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


def activities_target(cutin: Tuple[int, float, float], data_handler: DataHandler,
                      target_ngrams: NGram) -> ActivitiesResult:
    """ Extract the lateral and longitudinal activities of the target vehicle.

    :param cutin: Information of the cut in: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param target_ngrams: The n-grams of the targets.
    """
    activities = []
    acts = []
    target = Actor(TARGET, properties=dict(id=cutin[0], filename=data_handler.filename),
                   name="target vehicle")
    i_start, i_end = target_ngrams.start_and_end_indices("longitudinal_activity", i_ngram=cutin[0],
                                                         i_start=cutin[1], i_end=cutin[2])
    start = None
    for i, j in zip(i_start, i_end):
        activity_label = target_ngrams.ngrams[cutin[0]].loc[i, "longitudinal_activity"]
        activity_category = \
            (DEC_TARGET if activity_label == LongitudinalActivity.DECELERATING.value
             else ACC_TARGET if activity_label == LongitudinalActivity.ACCELERATING.value
             else CRU_TARGET)
        data = data_handler.targets[cutin[0]].loc[i:j, ["vx", "dx"]]
        n_knots = min(4, len(data)//10)
        activity = Activity(activity_category,
                            activity_category.fit(np.array(data.index), data.values,
                                                  n_knots=n_knots),
                            start=i if start is None else start, end=j-i,
                            name=activity_category.name)
        start = activity.end  # Store end event for potential next activity.
        activities.append(activity)
        acts.append((target, activity))

    # Store the lateral activity of the target.
    data = data_handler.targets[cutin[0]].loc[cutin[1]:cutin[2]]
    fix_line_data(data_handler.targets[cutin[0]],
                  data_handler.targets[cutin[0]].loc[cutin[1], "lateral_activity"],
                  cutin[1], cutin[2])
    data = data["line_center"]
    if np.any(np.isnan(data)):
        raise ValueError("NaN data!!!")
    n_knots = min(4, len(data)//10)
    activity = Activity(LC_TARGET, LC_TARGET.fit(np.array(data.index), data.values,
                                                 n_knots=n_knots),
                        start=cutin[1], end=cutin[2]-cutin[1], name="left lane change")
    if data_handler.targets[cutin[0]].loc[cutin[1], "lateral_activity"] == "li":
        activity.name = "right lane change"
    activities.append(activity)
    acts.append((target, activity))
    return ActivitiesResult(actor=target, activities=activities, acts=acts)


def activities_ego(cutin: Tuple[int, float, float], data_handler: DataHandler,
                   ego_ngram: NGram) -> ActivitiesResult:
    """ Extract the lateral and longitudinal activities of the ego vehicle.

    :param cutin: Information of the cut in: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param ego_ngram: The n-gram of the ego vehicle.
    """
    activities = []
    acts = []
    for type_of_activity in ["host_lateral_activity", "host_longitudinal_activity"]:
        i_start, i_end = ego_ngram.start_and_end_indices(type_of_activity, i_start=cutin[1],
                                                         i_end=cutin[2])
        start = None
        for i, j in zip(i_start, i_end):
            activity_label = ego_ngram.ngram.loc[i, type_of_activity]
            if type_of_activity == "host_longitudinal_activity":
                activity_category = \
                    (DEC_EGO if activity_label == LongitudinalActivity.DECELERATING.value
                     else ACC_EGO if activity_label == LongitudinalActivity.ACCELERATING.value
                     else CRU_EGO)
            else:
                activity_category = LK_EGO
            data = data_handler.data.loc[i:j, "Host_vx"]
            n_knots = min(4, len(data)//10)
            activity = Activity(activity_category,
                                activity_category.fit(np.array(data.index), data.values,
                                                      n_knots=n_knots),
                                start=i if start is None else start, end=j - i,
                                name=activity_category.name)
            start = activity.end  # Store end event for potential next activity.
            activities.append(activity)
            acts.append((EGO, activity))
    return ActivitiesResult(actor=EGO, activities=activities, acts=acts)


def process_cutin(cutin: Tuple[int, float, float], data_handler: DataHandler, target_ngrams: NGram,
                  ego_ngram: NGram, database: DocumentManagement) -> None:
    """ Process one cut in and add the data to the database.

    :param cutin: Information of the cut in: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param target_ngrams: The n-grams of the targets.
    :param ego_ngram: The n-gram of the ego vehicle.
    :param database: Database structure for storing the scenarios.
    """
    # Create the scenario.
    scenario = Scenario(start=cutin[1], end=cutin[2], physical_elements=[STAT])

    # Store the longitudinal activities of the target vehicle.
    target_acts = activities_target(cutin, data_handler, target_ngrams)
    scenario.set_actors([EGO, target_acts.actor])

    # Store the lateral and longitudinal activities of the ego vehicle.
    ego_acts = activities_ego(cutin, data_handler, ego_ngram)

    # Write everything to the database.
    scenario.set_activities(target_acts.activities+ego_acts.activities)
    scenario.set_acts(target_acts.acts+ego_acts.acts)
    database.add_item(scenario, include_attributes=True)


def process_file(path: str, database: DocumentManagement):
    """ Process a single HDF5 file.

    :param path: Path of the file with the n-grams.
    :param database: Database structure for storing the scenarios.
    """
    # Extract the cut ins.
    target_ngrams = NGram(FIELDNAMES_TARGET, METADATA_TARGET)
    target_ngrams.from_hdf(path, "targets")
    ego_ngram = NGram(FIELDNAMES_EGO, METADATA_EGO)
    ego_ngram.from_hdf(path, "ego")
    cutins = extract_cutins(target_ngrams, ego_ngram)

    # Store each cut in.
    data_handler = DataHandler(os.path.join("data", "1_hdf5", os.path.basename(path)))
    for cutin in cutins:
        if cutin[2] - cutin[1] < 0.1:
            continue  # Skip very short scenarios.
        process_cutin(cutin, data_handler, target_ngrams, ego_ngram, database)


if __name__ == "__main__":
    # Instantiate the "database".
    DATABASE = DocumentManagement()

    # Loop through the files.
    FILENAMES = glob.glob(os.path.join("data", "4_ngrams", "*.hdf5"))
    for filename in tqdm(FILENAMES):
        process_file(filename, DATABASE)

    DATABASE.to_json(os.path.join("data", "5_scenarios", "cut_in_scenarios2.json"), indent=4)
