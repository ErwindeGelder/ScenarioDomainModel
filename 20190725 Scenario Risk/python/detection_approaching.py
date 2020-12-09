""" Detect all scenarios where the ego vehicle approaches another vehicle.

Creation date: 2020 08 12
Author(s): Erwin de Gelder

Modifications:
2020 12 08: Update based on updated domain model.
"""

import glob
import os
from typing import List, Tuple
from tqdm import tqdm
from domain_model import DocumentManagement
from activity_detector import LeadVehicle, LateralStateTarget, LongitudinalStateTarget, TargetNear
from detection_lead_braking import process_file, STAT_CATEGORY, STAT, EGO_CATEGORY, EGO, TARGET, \
    DEC_TARGET, CRU_TARGET, ACC_TARGET, LK_TARGET, LC_TARGET, DEC_EGO, CRU_EGO, ACC_EGO, LK_EGO, \
    LC_EGO
from ngram import NGram
from ngram_search import find_sequence


def extract_approaching(ego_ngram: NGram, target_ngrams: NGram) -> List[Tuple[int, float, float]]:
    """ Extract cut ins.

    :param ego_ngram: The n-gram of the ego vehicle (not used).
    :param target_ngrams: The n-gram of the target vehicles.
    :return: A list of (itarget, start_braking, end_braking).
    """
    _ = ego_ngram
    # Define the tags for the ego-braking scenario.
    target_tags = [dict(lead_vehicle=[LeadVehicle.NOLEAD.value],
                        lateral_state=[LateralStateTarget.SAME.value],
                        longitudinal_state=[LongitudinalStateTarget.FRONT.value],
                        near=[TargetNear.FAR.value]),
                   dict(lead_vehicle=[LeadVehicle.LEAD.value],
                        lateral_state=[LateralStateTarget.SAME.value],
                        longitudinal_state=[LongitudinalStateTarget.FRONT.value],
                        near=[TargetNear.NEAR.value])]

    # Extract the scenarios where the target vehicle is decelerating.
    scenarios = []
    for i, target in enumerate(target_ngrams.ngrams):
        search = find_sequence((target,), (target_tags,))
        while search.is_found:
            if search.t_end - search.t_start > 0.1:  # Skip very short scenarios.
                scenarios.append((i, search.t_start, search.t_end))
            search = find_sequence((target,), (target_tags,), t_start=search.t_end)

    return scenarios


if __name__ == "__main__":
    # Instantiate the "database".
    DATABASE = DocumentManagement()

    # Add standard stuff.
    for item in [STAT_CATEGORY, STAT, EGO_CATEGORY, EGO, TARGET,
                 DEC_TARGET, CRU_TARGET, ACC_TARGET, LK_TARGET, LC_TARGET,
                 DEC_EGO, CRU_EGO, ACC_EGO, LK_EGO, LC_EGO]:
        DATABASE.add_item(item, include_attributes=True)

    # Loop through the files.
    FILENAMES = glob.glob(os.path.join("data", "4_ngrams", "*.hdf5"))
    N_SCENARIOS = 0
    for filename in tqdm(FILENAMES):
        N_SCENARIOS += process_file(filename, DATABASE, extract_approaching)
    print("Number of scenarios: {:d}".format(N_SCENARIOS))

    FOLDER = os.path.join("data", "5_scenarios")
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    DATABASE.to_json(os.path.join(FOLDER, "approaching_vehicle2.json"), indent=4)
