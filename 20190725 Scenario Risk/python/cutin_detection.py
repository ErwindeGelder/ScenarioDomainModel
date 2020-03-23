""" Detect all cut ins and store them in a kind-of database.

Creation date: 2020 03 23
Author(s): Erwin de Gelder

Modifications:
"""

import glob
import os
from typing import List, NamedTuple, Tuple
import numpy as np
from tqdm import tqdm
from domain_model import MultiBSplines, BSplines, ActivityCategory, StateVariable, Tag, \
    DetectedActivity, StaticEnvironmentCategory, StaticEnvironment, Region, ActorCategory, Actor, \
    VehicleType, EgoVehicle, Scenario
from activity_detector import LateralActivityTarget, LateralActivityHost, LeadVehicle, \
    LongitudinalActivity
from create_ngrams import FIELDNAMES_TARGET, FIELDNAMES_EGO, METADATA_TARGET, METADATA_EGO
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
LC_TARGET = ActivityCategory(BSplines(), StateVariable.LAT_TARGET, name="lane change target",
                             tags=[Tag.VehicleLateralActivity_ChangingLane])
LK_EGO = ActivityCategory(BSplines(), StateVariable.LATERAL_POSITION, name="lane keeping ego",
                          tags=[Tag.VehicleLateralActivity_GoingStraight])
STAT_CATEGORY = StaticEnvironmentCategory(Region.EU_WestCentral, description="Highway",
                                          name="Highway", tags=[Tag.RoadLayout_Straight])
STAT = StaticEnvironment(STAT_CATEGORY)
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
    for i, j in zip(i_start, i_end):
        activity_label = target_ngrams.ngrams[cutin[0]].loc[i, "longitudinal_activity"]
        activity_category = \
            (DEC_TARGET if activity_label == LongitudinalActivity.DECELERATING.value
             else ACC_TARGET if activity_label == LongitudinalActivity.ACCELERATING.value
             else CRU_TARGET)
        data = data_handler.targets[cutin[0]].loc[i:j, ["vx", "dx"]]
        activity = DetectedActivity(activity_category, i, j-i,
                                    activity_category.fit(np.array(data.index), data.values),
                                    name=activity_category.name)
        activities.append(activity)
        acts.append((target, activity, i))

    # Store the lateral activity of the target.
    data = data_handler.targets[cutin[0]].loc[cutin[1]:cutin[2], "dy"]
    activity = DetectedActivity(LC_TARGET, cutin[1], cutin[2]-cutin[1],
                                LC_TARGET.fit(np.array(data.index), data.values),
                                name="lane change target")
    activities.append(activity)
    acts.append((target, activity, cutin[1]))
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
            activity = DetectedActivity(activity_category, i, j - i,
                                        activity_category.fit(np.array(data.index), data.values),
                                        name=activity_category.name)
            activities.append(activity)
            acts.append((EGO, activity, i))
    return ActivitiesResult(actor=EGO, activities=activities, acts=acts)


def process_cutin(cutin: Tuple[int, float, float], data_handler: DataHandler, target_ngrams: NGram,
                  ego_ngram: NGram, database: DataBaseEmulator) -> None:
    """ Process one cut in and add the data to the database.

    :param cutin: Information of the cut in: (target_id, start_time, end_time).
    :param data_handler: Handler for the data.
    :param target_ngrams: The n-grams of the targets.
    :param ego_ngram: The n-gram of the ego vehicle.
    :param database: Database structure for storing the scenarios.
    """
    # Create the scenario.
    scenario = Scenario(cutin[1], cutin[2], STAT)

    # Store the longitudinal activities of the target vehicle.
    target_acts = activities_target(cutin, data_handler, target_ngrams)
    database.add_item(target_acts.actor)
    scenario.set_actors([EGO, target_acts.actor])

    # Store the lateral and longitudinal activities of the ego vehicle.
    ego_acts = activities_ego(cutin, data_handler, ego_ngram)

    # Write everything to the database.
    for activity in target_acts.activities+ego_acts.activities:
        database.add_item(activity)
    scenario.set_activities(target_acts.activities+ego_acts.activities)
    scenario.set_acts(target_acts.acts+ego_acts.acts)
    database.add_item(scenario)


def process_file(path: str, database: DataBaseEmulator):
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
        process_cutin(cutin, data_handler, target_ngrams, ego_ngram, database)


# if __name__ == "__main__":
# Instantiate the "database".
DATABASE = DataBaseEmulator()

# Add standard stuff.
for item in [DEC_TARGET, ACC_TARGET, CRU_TARGET, LC_TARGET, STAT_CATEGORY, STAT, TARGET,
             EGO_CATEGORY, EGO, DEC_EGO, ACC_EGO, CRU_EGO, LK_EGO]:
    DATABASE.add_item(item)

# Loop through the files.
FILENAMES = glob.glob(os.path.join("data", "4_ngrams", "*.hdf5"))
for filename in tqdm(FILENAMES):
    process_file(filename, DATABASE)
    break

DATABASE.to_json(os.path.join("data", "5_cutin_scenarios", "database.json"), indent=4)
