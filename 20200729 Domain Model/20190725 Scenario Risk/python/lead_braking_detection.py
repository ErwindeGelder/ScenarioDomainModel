""" Detect all lead braking activities and store them in a kind-of database.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
"""

import glob
import os
from typing import List, NamedTuple, Tuple
import numpy as np
from tqdm import tqdm
from domain_model import BSplines, ActivityCategory, StateVariable, Tag, \
    DetectedActivity, StaticEnvironmentCategory, StaticEnvironment, Region, ActorCategory, Actor, \
    VehicleType, EgoVehicle, Scenario
from activity_detector import LongitudinalActivity
from create_ngrams import FIELDNAMES_EGO, METADATA_EGO
from databaseemulator import DataBaseEmulator
from data_handler import DataHandler
from ngram import NGram
from ngram_search import find_sequence


ActivitiesResult = NamedTuple("ActivitiesResult",
                              [("actor", Actor), ("activities", List[DetectedActivity]),
                               ("acts", List[Tuple[Actor, DetectedActivity, float]])])


DEC_EGO = ActivityCategory(BSplines(), StateVariable.SPEED, name="deceleration ego",
                           tags=[Tag.VehicleLongitudinalActivity_DrivingForward_Braking])
STAT_CATEGORY = StaticEnvironmentCategory(Region.EU_WestCentral, description="Highway",
                                          name="Highway", tags=[Tag.RoadLayout_Straight])
STAT = StaticEnvironment(STAT_CATEGORY)
EGO_CATEGORY = ActorCategory(VehicleType.CategoryM_PassengerCar, name="ego vehicle category")
EGO = EgoVehicle(EGO_CATEGORY, name="ego vehicle")


def extract_ego_braking(ego_ngram: NGram) -> List[Tuple[float, float]]:
    """ Extract cut ins.

    :param ego_ngram: The n-gram of the ego vehicle.
    :return: A list of (start_braking, end_braking).
    """
    # Define the tags for the ego-braking scenario.
    ego_tags = [dict(host_longitudinal_activity=[LongitudinalActivity.DECELERATING.value])]

    # Extract the ego braking (=decelerating).
    braking = []
    search = find_sequence((ego_ngram.ngram,), (ego_tags,))
    while search.is_found:
        braking.append((search.t_start, search.t_end))
        search = find_sequence((ego_ngram.ngram,), (ego_tags,), t_start=search.t_end+5)
    return braking


def activities_ego(braking: Tuple[float, float], data_handler: DataHandler) -> ActivitiesResult:
    """ Extract the lateral and longitudinal activities of the ego vehicle.

    :param braking: Information of the braking: (start_time, end_time).
    :param data_handler: Handler for the data.
    """
    data = data_handler.data.loc[braking[0]:braking[1], "Host_vx"]
    data = data[data.values > 0]
    if len(data) < 50:
        return ActivitiesResult(actor=EGO, activities=[], acts=[])
    n_knots = min(4, len(data)//10)
    try:
        activity = DetectedActivity(DEC_EGO, data.index[0], data.index[-1] - data.index[0],
                                    DEC_EGO.fit(np.array(data.index), data.values,
                                                options=dict(n_knots=n_knots)),
                                    name="Ego braking in file {:s}".format(data_handler.filename))
    except ValueError:
        # Why does this error occur? Very strange...
        return ActivitiesResult(actor=EGO, activities=[], acts=[])
    return ActivitiesResult(actor=EGO, activities=[activity], acts=[(EGO, activity, braking[0])])


def process_ego_braking(braking: Tuple[float, float], data_handler: DataHandler,
                        database: DataBaseEmulator) -> None:
    """ Process one cut in and add the data to the database.

    :param braking: Information of the ego braking: (start_time, end_time).
    :param data_handler: Handler for the data.
    :param database: Database structure for storing the scenarios.
    """
    # Store the longitudinal activity of the ego vehicle.
    ego_acts = activities_ego(braking, data_handler)
    if not ego_acts.acts:
        return

    # Write everything to the database.
    for activity in ego_acts.activities:
        database.add_item(activity)
    scenario = Scenario(ego_acts.activities[0].tstart, ego_acts.activities[0].tend, STAT)
    scenario.set_actors([EGO])
    scenario.set_activities(ego_acts.activities)
    scenario.set_acts(ego_acts.acts)
    database.add_item(scenario)


def process_file(path: str, database: DataBaseEmulator):
    """ Process a single HDF5 file.

    :param path: Path of the file with the n-grams.
    :param database: Database structure for storing the scenarios.
    """
    # Extract the deceleration activities of the ego vehicle.
    ego_ngram = NGram(FIELDNAMES_EGO, METADATA_EGO)
    ego_ngram.from_hdf(path, "ego")
    brakings = extract_ego_braking(ego_ngram)

    # Store each cut in.
    data_handler = DataHandler(os.path.join("data", "1_hdf5", os.path.basename(path)))
    for braking in brakings:
        if braking[1] - braking[0] < 0.1:
            continue  # Skip very short scenarios.
        process_ego_braking(braking, data_handler, database)
    return len(brakings)


if __name__ == "__main__":
    # Instantiate the "database".
    DATABASE = DataBaseEmulator()

    # Add standard stuff.
    for item in [STAT_CATEGORY, STAT, EGO_CATEGORY, EGO, DEC_EGO]:
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
    DATABASE.to_json(os.path.join(FOLDER, "ego_braking.json"), indent=4)
